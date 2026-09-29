#include "kernel_operator.h"

// NV12 letterbox padding must stay a neutral BT.601 limited-range gray: Y=0 with
// U=V=0 would give the static AIPP a saturated border instead of the gray the
// detector was trained with. The actual values arrive as kernel arguments so
// they can be tuned from the model config.
//
// HOW THE WORK IS ARRANGED
//
// A tile covers `kChunk` output bytes of one plane row, so everything it reads
// lies in a couple of contiguous source rows. Those rows are copied into UB once
// and the chunk is computed from there.
//
// The earlier version sampled the source straight from GM with `GetValue`, and
// for chroma that meant 24 scalar reads for every output byte: four luma
// positions, three channels each, two rows each. On this part a scalar GM read
// costs about 40 ns and is latency bound, so those reads dominated the step;
// staging replaces them with a few bursts per tile and turns the inner reads
// into UB accesses.
//
// The arithmetic is unchanged, operation for operation - including the order the
// four chroma samples are summed and the order the YUV matrix is evaluated - so
// results stay bit-identical to the version that read from GM.

// One byte of a staged BGR row.
__aicore__ inline int32_t ReadBgr(const AscendC::LocalTensor<uint8_t>& row,
                                  int32_t byte_offset)
{
    return static_cast<int32_t>(row.GetValue(byte_offset));
}

// Maps an output column to the source columns it interpolates and the weight of
// the second one. Same expression the earlier kernel evaluated inline.
__aicore__ inline void MapX(int32_t output_x, int32_t pad_left,
                            int32_t source_width, int32_t resized_width,
                            int32_t* x0, int32_t* x1, float* weight)
{
    float source_x = (static_cast<float>(output_x - pad_left) + 0.5f) *
                         static_cast<float>(source_width) /
                         static_cast<float>(resized_width) -
                     0.5f;
    source_x = source_x < 0.0f ? 0.0f : source_x;
    source_x = source_x > source_width - 1
                   ? static_cast<float>(source_width - 1)
                   : source_x;
    *x0 = static_cast<int32_t>(source_x);
    *x1 = *x0 + 1 < source_width ? *x0 + 1 : *x0;
    *weight = source_x - static_cast<float>(*x0);
}

// Maps an output row to the source rows it interpolates and the weight of the
// second one. Same expression as above, on the vertical axis.
__aicore__ inline void MapY(int32_t output_y, int32_t pad_top,
                            int32_t source_height, int32_t resized_height,
                            int32_t* y0, int32_t* y1, float* weight)
{
    float source_y = (static_cast<float>(output_y - pad_top) + 0.5f) *
                         static_cast<float>(source_height) /
                         static_cast<float>(resized_height) -
                     0.5f;
    source_y = source_y < 0.0f ? 0.0f : source_y;
    source_y = source_y > source_height - 1
                   ? static_cast<float>(source_height - 1)
                   : source_y;
    *y0 = static_cast<int32_t>(source_y);
    *y1 = *y0 + 1 < source_height ? *y0 + 1 : *y0;
    *weight = source_y - static_cast<float>(*y0);
}

// Bilinear blend of one channel out of two staged rows, both starting at
// `span_start` inside their row.
__aicore__ inline float SampleBgr(const AscendC::LocalTensor<uint8_t>& rows,
                                  int32_t row0, int32_t row1, int32_t x0_byte,
                                  int32_t x1_byte, int32_t channel, float weight_x,
                                  float weight_y)
{
    const float top =
        static_cast<float>(ReadBgr(rows, row0 + x0_byte + channel)) *
            (1.0f - weight_x) +
        static_cast<float>(ReadBgr(rows, row0 + x1_byte + channel)) * weight_x;
    const float bottom =
        static_cast<float>(ReadBgr(rows, row1 + x0_byte + channel)) *
            (1.0f - weight_x) +
        static_cast<float>(ReadBgr(rows, row1 + x1_byte + channel)) * weight_x;
    return top * (1.0f - weight_y) + bottom * weight_y;
}

__aicore__ inline float ConvertYuv(float blue, float green, float red,
                                   int32_t component)
{
    float value = 0.0f;
    if (component == 0) {
        value = 16.0f + 0.257f * red + 0.504f * green + 0.098f * blue;
    } else if (component == 1) {
        value = 128.0f - 0.148f * red - 0.291f * green + 0.439f * blue;
    } else {
        value = 128.0f + 0.439f * red - 0.368f * green - 0.071f * blue;
    }
    value = value < 0.0f ? 0.0f : value;
    return value > 255.0f ? 255.0f : value;
}

// One sample of the YUV image at an output position, from two staged rows. The
// horizontal map is not computed here: it depends on the output column and
// nothing else, so the caller reads it out of the table BuildColumnMap filled.
// `x0_byte` and `x1_byte` are already byte offsets into the staged rows.
__aicore__ inline float SampleYuv(const AscendC::LocalTensor<uint8_t>& rows,
                                  int32_t row0, int32_t row1, int32_t x0_byte,
                                  int32_t x1_byte, float weight_x,
                                  float weight_y, int32_t component)
{
    const float blue = SampleBgr(rows, row0, row1, x0_byte, x1_byte, 0,
                                 weight_x, weight_y);
    const float green = SampleBgr(rows, row0, row1, x0_byte, x1_byte, 1,
                                  weight_x, weight_y);
    const float red = SampleBgr(rows, row0, row1, x0_byte, x1_byte, 2,
                                weight_x, weight_y);
    return ConvertYuv(blue, green, red, component);
}

// Fills the horizontal map for one plane into three tables.
//
// MapX answers a question that only depends on the output column, yet the inner
// loops asked it once per output byte: a float divide, two clamps, a truncation
// and a compare. On this part that is the bulk of the work per byte, and it is
// the same answer every time, so it is cheaper to answer it once per launch per
// core than several hundred thousand times.
//
// Entries are stored the way the inner loops consume them: byte offsets into a
// staged row, already scaled by the plane's bytes per sample, so a lookup only
// has to subtract `span_start` and add the component index. `unit_count` is the
// number of distinct output columns in the plane's own resolution, and the
// geometry arguments are already expressed in that resolution too.
//
// The arithmetic is the same expression MapX evaluates, value for value, so the
// result does not change - only how often it is worked out.
__aicore__ inline void BuildColumnMap(
    AscendC::LocalTensor<int32_t> x0_bytes,
    AscendC::LocalTensor<int32_t> x1_bytes,
    AscendC::LocalTensor<float> weights, int32_t base, int32_t unit_count,
    int32_t pad_left, int32_t source_width, int32_t resized_width,
    int32_t bytes_per_sample)
{
    for (int32_t unit = 0; unit < unit_count; ++unit) {
        int32_t x0 = 0;
        int32_t x1 = 0;
        float weight = 0.0f;
        MapX(unit, pad_left, source_width, resized_width, &x0, &x1, &weight);
        x0_bytes.SetValue(base + unit, x0 * bytes_per_sample);
        x1_bytes.SetValue(base + unit, x1 * bytes_per_sample);
        weights.SetValue(base + unit, weight);
    }
}

// Byte offset of the first source column a chunk reads, and how much of the row
// to stage. Returns false when the chunk touches no image pixel.
__aicore__ inline bool SpanFor(int32_t first_x, int32_t last_x, int32_t pad_left,
                               int32_t source_width, int32_t resized_width,
                               int32_t row_bytes, int32_t* span_start,
                               int32_t* span_need)
{
    if (last_x < first_x) {
        return false;
    }
    int32_t x0 = 0;
    int32_t x1 = 0;
    float weight = 0.0f;
    MapX(first_x, pad_left, source_width, resized_width, &x0, &x1, &weight);
    // Start on a burst boundary. A chunk never reads before its first column, so
    // backing up only fetches bytes nobody looks at, and it keeps the source
    // side of the copy aligned.
    *span_start = (x0 * 3) & ~31;
    MapX(last_x, pad_left, source_width, resized_width, &x0, &x1, &weight);
    *span_need = (x1 + 1) * 3 - *span_start;
    if (*span_need <= 0 || *span_start + *span_need > row_bytes) {
        return false;
    }
    return true;
}

// Copies `span_need` bytes from `byte_base` into `dst`, in whole 32 byte bursts
// with the remainder fetched one byte at a time: DataCopy only moves whole
// bursts, and a row is not padded out to one. A row is never over-read, because
// source_width is not always the image width and the last row would otherwise
// leave the buffer.
__aicore__ inline void StageRow(const AscendC::GlobalTensor<uint8_t>& image,
                                const AscendC::LocalTensor<uint8_t>& dst,
                                int32_t byte_base, int32_t span_start,
                                int32_t span_need, int32_t row_bytes)
{
    int32_t burst_end = (span_start + span_need) & ~31;
    if (burst_end > row_bytes) {
        burst_end = row_bytes & ~31;
    }
    const int32_t bulk = burst_end > span_start ? burst_end - span_start : 0;
    if (bulk > 0) {
        AscendC::DataCopy(dst, image[byte_base + span_start], bulk);
    }
    for (int32_t byte = bulk; byte < span_need; ++byte) {
        dst.SetValue(byte, image.GetValue(byte_base + span_start + byte));
    }
}

// Stores a staged chunk of `length` bytes, using whole bursts where the
// destination lines up and single element writes where it does not.
__aicore__ inline void StoreChunk(AscendC::GlobalTensor<uint8_t>& output,
                                  const AscendC::LocalTensor<uint8_t>& chunk,
                                  int32_t base, int32_t length)
{
    const int32_t burst = (base & 31) == 0 ? (length & ~31) : 0;
    if (burst > 0) {
        AscendC::DataCopy(output[base], chunk, burst);
    }
    for (int32_t index = burst; index < length; ++index) {
        output.SetValue(base + index, chunk.GetValue(index));
    }
}

// The positions in [0, length) that land inside the image, as a half-open
// chunk-relative range. The ends are given inclusive.
//
// The map is monotonic in x, so those positions are always one contiguous run:
// find its ends up front and the loop that follows needs no bounds test at all,
// which is worth more than it looks because the test would sit on every output
// byte. Note that the bounds are in *position* space - for luma a position is
// its own column, but a chroma byte at odd position p only reaches column p-1,
// so a column bound is not a position bound.
__aicore__ inline void InsideRun(int32_t chunk_x, int32_t length,
                                 int32_t first_position, int32_t last_position,
                                 int32_t* begin, int32_t* end)
{
    int32_t b = first_position > chunk_x ? first_position - chunk_x : 0;
    if (b > length) {
        b = length;
    }
    int32_t e = last_position + 1 > chunk_x ? last_position + 1 - chunk_x : 0;
    if (e > length) {
        e = length;
    }
    if (e < b) {
        e = b;
    }
    *begin = b;
    *end = e;
}

extern "C" __global__ __aicore__ void yolo_preprocess(
    GM_ADDR input, GM_ADDR output, int32_t input_height, int32_t input_width,
    int32_t output_height, int32_t output_width, int32_t resized_height,
    int32_t resized_width, int32_t pad_top, int32_t pad_left,
    int32_t source_height, int32_t source_width, int32_t pad_luma,
    int32_t pad_chroma, int32_t block_count)
{
    // Output bytes per tile. Staging a whole row would need the whole source row
    // in UB, which is fine for 640 but not for an arbitrarily wide input;
    // chunking keeps the staged span proportional to the chunk instead.
    constexpr int32_t kChunk = 128;

    AscendC::GlobalTensor<uint8_t> input_tensor;
    AscendC::GlobalTensor<uint8_t> output_tensor;
    AscendC::TPipe pipe;
    input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(input),
                                 input_height * input_width * 3);
    const int32_t y_plane_size = output_height * output_width;
    output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output),
                                  y_plane_size * 3 / 2);

    // A chroma tile reads four source rows: two for each of the two luma rows it
    // averages. Luma tiles use the first two slots.
    const int32_t row_bytes = input_width * 3;
    const int32_t span_bytes = ((source_width * 3 + 31) / 32) * 32 + 32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> row_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> chunk_buf;
    // One entry per output column. Both planes sample the BGR image in luma
    // resolution - chroma is averaged over a 2x2 block of converted luma
    // samples - so one table covers both passes.
    const int32_t map_units = output_width;
    const int32_t map_bytes = ((map_units * 4) + 31) / 32 * 32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> map_x0_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> map_x1_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> map_weight_buf;
    pipe.InitBuffer(row_buf, span_bytes * 4);
    pipe.InitBuffer(chunk_buf, kChunk);
    pipe.InitBuffer(map_x0_buf, map_bytes);
    pipe.InitBuffer(map_x1_buf, map_bytes);
    pipe.InitBuffer(map_weight_buf, map_bytes);
    AscendC::LocalTensor<uint8_t> rows = row_buf.Get<uint8_t>();
    AscendC::LocalTensor<uint8_t> chunk = chunk_buf.Get<uint8_t>();
    AscendC::LocalTensor<int32_t> map_x0 = map_x0_buf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> map_x1 = map_x1_buf.Get<int32_t>();
    AscendC::LocalTensor<float> map_weight = map_weight_buf.Get<float>();
    BuildColumnMap(map_x0, map_x1, map_weight, 0, map_units, pad_left,
                   source_width, resized_width, 3);

    const float pad_luma_value = static_cast<float>(pad_luma);
    const float pad_chroma_value = static_cast<float>(pad_chroma);
    const int32_t chunks_per_row = (output_width + kChunk - 1) / kChunk;
    const int32_t luma_tiles = output_height * chunks_per_row;
    const int32_t total_tiles = luma_tiles + (output_height / 2) * chunks_per_row;
    const int32_t inside_x_end = pad_left + resized_width - 1;

    for (int32_t tile = AscendC::GetBlockIdx(); tile < total_tiles;
         tile += block_count) {
        const bool luma = tile < luma_tiles;
        const int32_t plane_tile = luma ? tile : tile - luma_tiles;
        const int32_t plane_row = plane_tile / chunks_per_row;
        const int32_t chunk_x =
            (plane_tile - plane_row * chunks_per_row) * kChunk;
        int32_t length = output_width - chunk_x;
        if (length > kChunk) {
            length = kChunk;
        }
        const int32_t output_y = luma ? plane_row : plane_row * 2;
        const int32_t resized_y = output_y - pad_top;

        // A luma sample only needs its own row inside; a chroma sample averages
        // two luma rows and needs both of them.
        const bool rows_inside =
            luma ? (resized_y >= 0 && resized_y < resized_height)
                 : (resized_y >= 0 && resized_y + 1 < resized_height);
        int32_t y0 = 0;
        int32_t y1 = 0;
        int32_t y2 = 0;
        int32_t y3 = 0;
        float weight_y0 = 0.0f;
        float weight_y1 = 0.0f;
        if (rows_inside) {
            MapY(output_y, pad_top, source_height, resized_height, &y0, &y1,
                 &weight_y0);
            if (!luma) {
                MapY(output_y + 1, pad_top, source_height, resized_height, &y2,
                     &y3, &weight_y1);
            }
        }

        // Stage the source span this chunk reads. The mapping is monotonic in x,
        // so the span runs from the first to the last column it touches. A
        // chroma sample also reads the next column along.
        int32_t span_start = 0;
        int32_t span_need = 0;
        const bool have_span =
            rows_inside &&
            SpanFor(chunk_x > pad_left ? chunk_x : pad_left,
                    luma ? (chunk_x + length - 1 < inside_x_end
                                ? chunk_x + length - 1
                                : inside_x_end)
                         : (chunk_x + length < inside_x_end
                                ? chunk_x + length
                                : inside_x_end),
                    pad_left, source_width, resized_width, row_bytes, &span_start,
                    &span_need);

        if (have_span) {
            StageRow(input_tensor, rows, y0 * row_bytes, span_start, span_need,
                     row_bytes);
            StageRow(input_tensor, rows[span_bytes], y1 * row_bytes, span_start,
                     span_need, row_bytes);
            if (!luma) {
                StageRow(input_tensor, rows[span_bytes * 2], y2 * row_bytes,
                         span_start, span_need, row_bytes);
                StageRow(input_tensor, rows[span_bytes * 3], y3 * row_bytes,
                         span_start, span_need, row_bytes);
            }
        }

        // A chunk is three runs: the border to the left of the image, the image
        // itself, and the border to its right. See InsideRun for why the middle
        // run can be delimited before the loop starts; the two border runs just
        // write a constant.
        int32_t image_begin = 0;
        int32_t image_end = 0;
        const uint8_t pad_luma_byte = static_cast<uint8_t>(
            static_cast<int32_t>(pad_luma_value + 0.5f));
        const uint8_t pad_chroma_byte = static_cast<uint8_t>(
            static_cast<int32_t>(pad_chroma_value + 0.5f));

        if (luma) {
            // A luma byte at column x is inside when x is in
            // [pad_left, pad_left + resized_width), and its position is x.
            InsideRun(chunk_x, length, pad_left, pad_left + resized_width - 1,
                      &image_begin, &image_end);
            if (!have_span) {
                image_begin = 0;
                image_end = 0;
            }
            for (int32_t index = 0; index < image_begin; ++index) {
                chunk.SetValue(index, pad_luma_byte);
            }
            for (int32_t index = image_begin; index < image_end; ++index) {
                const int32_t output_x = chunk_x + index;
                const float value = SampleYuv(
                    rows, 0, span_bytes, map_x0.GetValue(output_x) - span_start,
                    map_x1.GetValue(output_x) - span_start,
                    map_weight.GetValue(output_x), weight_y0, 0);
                chunk.SetValue(index, static_cast<uint8_t>(
                                          static_cast<int32_t>(value + 0.5f)));
            }
            for (int32_t index = image_end; index < length; ++index) {
                chunk.SetValue(index, pad_luma_byte);
            }
            StoreChunk(output_tensor, chunk, output_y * output_width + chunk_x,
                       length);
        } else {
            // A chroma byte at position p reads columns p-1 and p (p odd), and
            // both have to be inside: p - 1 >= pad_left and p < pad_left +
            // resized_width. So the columns are bounded first and squared up to
            // even, then turned back into the positions that cover them - the
            // last inside position is one past the last inside column, because
            // column x is shared by positions x and x + 1.
            const int32_t chroma_last_x = pad_left + resized_width - 2;
            if (chroma_last_x < pad_left) {
                image_begin = 0;
                image_end = 0;
            } else {
                InsideRun(chunk_x, length, (pad_left + 1) & ~1,
                          (chroma_last_x & ~1) + 1, &image_begin, &image_end);
            }
            if (!have_span) {
                image_begin = 0;
                image_end = 0;
            }
            for (int32_t index = 0; index < image_begin; ++index) {
                chunk.SetValue(index, pad_chroma_byte);
            }
            for (int32_t index = image_begin; index < image_end; ++index) {
                const int32_t position = chunk_x + index;
                const int32_t output_x = position & ~1;
                // Both columns the 2x2 block covers share every stage of the
                // conversion except this lookup, so it is read once each.
                const int32_t component = (position & 1) ? 2 : 1;
                const int32_t left = map_x0.GetValue(output_x) - span_start;
                const int32_t left1 = map_x1.GetValue(output_x) - span_start;
                const float left_weight = map_weight.GetValue(output_x);
                const int32_t right = map_x0.GetValue(output_x + 1) - span_start;
                const int32_t right1 =
                    map_x1.GetValue(output_x + 1) - span_start;
                const float right_weight = map_weight.GetValue(output_x + 1);
                const float value =
                    0.25f *
                    (SampleYuv(rows, 0, span_bytes, left, left1, left_weight,
                               weight_y0, component) +
                     SampleYuv(rows, 0, span_bytes, right, right1, right_weight,
                               weight_y0, component) +
                     SampleYuv(rows, span_bytes * 2, span_bytes * 3, left, left1,
                               left_weight, weight_y1, component) +
                     SampleYuv(rows, span_bytes * 2, span_bytes * 3, right,
                               right1, right_weight, weight_y1, component));
                chunk.SetValue(index, static_cast<uint8_t>(
                                          static_cast<int32_t>(value + 0.5f)));
            }
            for (int32_t index = image_end; index < length; ++index) {
                chunk.SetValue(index, pad_chroma_byte);
            }
            StoreChunk(output_tensor, chunk,
                       y_plane_size + plane_row * output_width + chunk_x,
                       length);
        }
    }
}

// Letterbox for a caller that already holds NV12.
//
// Nothing on this path converts colour: the source planes are resampled as they
// are and written into the canvas the AIPP model reads. That is the whole point -
// a camera or DVPP pipeline already produces the form the model wants, so the
// BGR round trip the `IMAGE` path performs would only add a conversion and lose
// precision doing it.
//
// Luma is bilinear sampled with the usual letterbox mapping. Chroma is sampled
// with the same mapping in half resolution space, because that is where the
// source chroma lives: canvas chroma column c covers canvas luma columns 2c and
// 2c+1, so it maps to source chroma column
// MapX(c, pad_left / 2, source_width / 2, resized_width / 2). The two passes are
// therefore the same code over a different plane: one luma byte per unit, or one
// chroma column of two interleaved bytes per unit, and a different set of
// geometry values.
//
// The border takes the configured pad values, which are already in the limited
// range BT.601 domain the source is in.
extern "C" __global__ __aicore__ void yolo_yuv_preprocess(
    GM_ADDR input, GM_ADDR output, int32_t input_height, int32_t input_width,
    int32_t output_height, int32_t output_width, int32_t resized_height,
    int32_t resized_width, int32_t pad_top, int32_t pad_left,
    int32_t source_height, int32_t source_width, int32_t pad_luma,
    int32_t pad_chroma, int32_t block_count)
{
    constexpr int32_t kChunk = 128;

    AscendC::GlobalTensor<uint8_t> input_tensor;
    AscendC::GlobalTensor<uint8_t> output_tensor;
    AscendC::TPipe pipe;
    input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(input),
                                 input_height * 3 / 2 * input_width);
    const int32_t y_plane_size = output_height * output_width;
    output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output),
                                  y_plane_size * 3 / 2);

    // Luma rows and interleaved chroma rows are both `input_width` bytes wide,
    // so one stride addresses the whole source frame.
    const int32_t row_bytes = input_width;
    const int32_t chroma_base = input_height * input_width;
    const int32_t span_bytes = ((source_width + 31) / 32) * 32 + 64;
    AscendC::TBuf<AscendC::TPosition::VECCALC> row_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> chunk_buf;
    // A unit is one luma byte, or one chroma column of two interleaved bytes, so
    // chroma needs half as many entries. Both planes' maps live in one table,
    // luma first, which keeps the inner loop free of a base offset test.
    const int32_t luma_units = output_width;
    const int32_t chroma_units = (output_width + 1) / 2;
    const int32_t map_bytes = (((luma_units + chroma_units) * 4) + 31) / 32 * 32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> map_x0_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> map_x1_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> map_weight_buf;
    pipe.InitBuffer(row_buf, span_bytes * 2);
    pipe.InitBuffer(chunk_buf, kChunk);
    pipe.InitBuffer(map_x0_buf, map_bytes);
    pipe.InitBuffer(map_x1_buf, map_bytes);
    pipe.InitBuffer(map_weight_buf, map_bytes);
    AscendC::LocalTensor<uint8_t> rows = row_buf.Get<uint8_t>();
    AscendC::LocalTensor<uint8_t> chunk = chunk_buf.Get<uint8_t>();
    AscendC::LocalTensor<int32_t> map_x0 = map_x0_buf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> map_x1 = map_x1_buf.Get<int32_t>();
    AscendC::LocalTensor<float> map_weight = map_weight_buf.Get<float>();
    BuildColumnMap(map_x0, map_x1, map_weight, 0, luma_units, pad_left,
                   source_width, resized_width, 1);
    BuildColumnMap(map_x0, map_x1, map_weight, luma_units, chroma_units,
                   pad_left / 2, source_width / 2, resized_width / 2, 2);

    const float pad_luma_value = static_cast<float>(pad_luma);
    const float pad_chroma_value = static_cast<float>(pad_chroma);
    const int32_t chunks_per_row = (output_width + kChunk - 1) / kChunk;
    const int32_t luma_tiles = output_height * chunks_per_row;
    const int32_t total_tiles =
        luma_tiles + (output_height / 2) * chunks_per_row;

    for (int32_t tile = AscendC::GetBlockIdx(); tile < total_tiles;
         tile += block_count) {
        const bool luma = tile < luma_tiles;
        const int32_t plane_tile = luma ? tile : tile - luma_tiles;
        const int32_t plane_row = plane_tile / chunks_per_row;
        const int32_t chunk_x =
            (plane_tile - plane_row * chunks_per_row) * kChunk;
        int32_t length = output_width - chunk_x;
        if (length > kChunk) {
            length = kChunk;
        }

        // The same geometry, expressed in the plane's own resolution. A unit is
        // one luma byte, or one chroma column of two interleaved bytes.
        const int32_t plane_pad_left = luma ? pad_left : pad_left / 2;
        const int32_t plane_pad_top = luma ? pad_top : pad_top / 2;
        const int32_t plane_source_height =
            luma ? source_height : source_height / 2;
        const int32_t plane_resized_height =
            luma ? resized_height : resized_height / 2;
        const int32_t plane_source_width = luma ? source_width : source_width / 2;
        const int32_t plane_resized_width =
            luma ? resized_width : resized_width / 2;
        const int32_t bytes_per_unit = luma ? 1 : 2;
        const int32_t first_unit =
            (luma ? chunk_x : chunk_x / 2) > plane_pad_left
                ? (luma ? chunk_x : chunk_x / 2)
                : plane_pad_left;
        const int32_t last_unit =
            luma ? chunk_x + length - 1 : (chunk_x + length - 1) / 2;
        const int32_t inside_end = plane_pad_left + plane_resized_width - 1;
        const int32_t last_inside =
            last_unit < inside_end ? last_unit : inside_end;
        const int32_t resized_row = plane_row - plane_pad_top;
        const bool row_inside = resized_row >= 0 &&
                                resized_row < plane_resized_height;

        int32_t row0 = 0;
        int32_t row1 = 0;
        float weight_y = 0.0f;
        int32_t span_start = 0;
        int32_t span_need = 0;
        if (row_inside && last_inside >= first_unit) {
            MapY(plane_row, plane_pad_top, plane_source_height,
                 plane_resized_height, &row0, &row1, &weight_y);
            int32_t u0 = 0;
            int32_t u1 = 0;
            float weight = 0.0f;
            MapX(first_unit, plane_pad_left, plane_source_width,
                 plane_resized_width, &u0, &u1, &weight);
            span_start = (u0 * bytes_per_unit) & ~31;
            MapX(last_inside, plane_pad_left, plane_source_width,
                 plane_resized_width, &u0, &u1, &weight);
            span_need = (u1 + 1) * bytes_per_unit - span_start;
        }

        if (span_need > 0) {
            const int32_t base = luma ? 0 : chroma_base;
            StageRow(input_tensor, rows, base + row0 * row_bytes, span_start,
                     span_need, row_bytes);
            StageRow(input_tensor, rows[span_bytes], base + row1 * row_bytes,
                     span_start, span_need, row_bytes);
        }

        // The plane differences are all shifts and masks rather than branches:
        // a luma unit is its own column, a chroma byte is one half of a two
        // byte pair, so its column is the position shifted down and the low bit
        // picks U or V.
        const int32_t unit_shift = luma ? 0 : 1;
        const int32_t component_mask = luma ? 0 : 1;
        const int32_t unit_base = luma ? 0 : luma_units;
        const float pad_value = luma ? pad_luma_value : pad_chroma_value;

        // A chunk is three runs: the border to the left of the image, the image
        // itself, and the border to its right. The map is monotonic in x, so the
        // image part is one contiguous range of positions and its extent is
        // known before the loop starts; splitting it out leaves the middle loop
        // with no bounds test and no branch at all, and lets the two border runs
        // write a constant.
        const int32_t inside_from = plane_pad_left << unit_shift;
        const int32_t inside_until =
            (plane_pad_left + plane_resized_width) << unit_shift;
        int32_t image_begin = chunk_x < inside_from ? inside_from - chunk_x : 0;
        if (image_begin > length) {
            image_begin = length;
        }
        int32_t image_end = chunk_x < inside_until ? inside_until - chunk_x : 0;
        if (image_end > length) {
            image_end = length;
        }
        if (image_end < image_begin) {
            image_end = image_begin;
        }
        if (span_need <= 0) {
            // No row staged: every position is border, so the middle run empties
            // and the two border runs cover the chunk.
            image_begin = 0;
            image_end = 0;
        }
        const uint8_t pad_byte = static_cast<uint8_t>(
            static_cast<int32_t>(pad_value + 0.5f));

        for (int32_t index = 0; index < image_begin; ++index) {
            chunk.SetValue(index, pad_byte);
        }
        for (int32_t index = image_begin; index < image_end; ++index) {
            const int32_t position = chunk_x + index;
            const int32_t unit = position >> unit_shift;
            const int32_t component = position & component_mask;
            const int32_t offset0 =
                map_x0.GetValue(unit_base + unit) + component - span_start;
            const int32_t offset1 =
                map_x1.GetValue(unit_base + unit) + component - span_start;
            const float weight_x = map_weight.GetValue(unit_base + unit);
            const float top = static_cast<float>(ReadBgr(rows, offset0)) *
                                  (1.0f - weight_x) +
                              static_cast<float>(ReadBgr(rows, offset1)) *
                                  weight_x;
            const float bottom =
                static_cast<float>(ReadBgr(rows, span_bytes + offset0)) *
                    (1.0f - weight_x) +
                static_cast<float>(ReadBgr(rows, span_bytes + offset1)) *
                    weight_x;
            const float value = top * (1.0f - weight_y) + bottom * weight_y;
            chunk.SetValue(index, static_cast<uint8_t>(
                                      static_cast<int32_t>(value + 0.5f)));
        }
        for (int32_t index = image_end; index < length; ++index) {
            chunk.SetValue(index, pad_byte);
        }

        StoreChunk(output_tensor, chunk,
                   (luma ? 0 : y_plane_size) + plane_row * output_width +
                       chunk_x,
                   length);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void yolo_yuv_preprocess_do(uint32_t block_dim, void* stream, uint8_t* input,
                            uint8_t* output, int32_t input_height,
                            int32_t input_width, int32_t output_height,
                            int32_t output_width, int32_t resized_height,
                            int32_t resized_width, int32_t pad_top,
                            int32_t pad_left, int32_t source_height,
                            int32_t source_width, int32_t pad_luma,
                            int32_t pad_chroma)
{
    yolo_yuv_preprocess<<<block_dim, nullptr, stream>>>(
        input, output, input_height, input_width, output_height, output_width,
        resized_height, resized_width, pad_top, pad_left, source_height,
        source_width, pad_luma, pad_chroma, static_cast<int32_t>(block_dim));
}
#endif

#ifndef ASCENDC_CPU_DEBUG
void yolo_preprocess_do(uint32_t block_dim, void* stream, uint8_t* input,
                       uint8_t* output, int32_t input_height,
                       int32_t input_width, int32_t output_height,
                       int32_t output_width, int32_t resized_height,
                       int32_t resized_width, int32_t pad_top,
                       int32_t pad_left, int32_t source_height,
                       int32_t source_width, int32_t pad_luma,
                       int32_t pad_chroma)
{
    yolo_preprocess<<<block_dim, nullptr, stream>>>(
        input, output, input_height, input_width, output_height, output_width,
        resized_height, resized_width, pad_top, pad_left, source_height,
        source_width, pad_luma, pad_chroma, static_cast<int32_t>(block_dim));
}
#endif
