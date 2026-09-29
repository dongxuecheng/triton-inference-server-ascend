#include "kernel_operator.h"

// Letterbox preprocessing for a model converted *without* AIPP.
//
// A model built with `--insert_op_conf=aipp_nv12_640.cfg` carries the colour
// conversion and the normalisation inside the AIPP unit, so it consumes the
// packed NV12 canvas that `preprocess/yolo_preprocess_kernel.cpp` writes. A
// model built without that option has no AIPP at all and consumes the ONNX
// input tensor directly, which for a YOLO export is FP32, RGB, planar (CHW) and
// scaled to [0, 1]. This kernel writes exactly that tensor.
//
// The geometry is deliberately identical to the AIPP path - same letterbox, same
// bilinear resize, same border gray - so the two chains differ only in how the
// colours are carried: here BGR goes straight to RGB, while the AIPP path takes
// a YUV round trip through the static AIPP matrix.
//
// HOW THE WORK IS ARRANGED
//
// A tile covers `kChunk` output pixels of one row, so everything it reads lies
// in two contiguous source rows. Those two rows are copied into UB once and the
// whole chunk is computed from there, all three planes included, because the
// staged rows are interleaved BGR and every plane reads the same bytes.
//
// The earlier version sampled the source straight from GM with `GetValue`, four
// scalar reads per output pixel. On this part a scalar GM read costs about 40 ns
// and is latency bound, so those millions of reads dominated the step; staging
// replaces them with a few bursts per tile.
//
// The arithmetic is unchanged, operation for operation, so results stay
// bit-identical to the version that read from GM. Only the way the bytes get
// under the ALU is different.

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

extern "C" __global__ __aicore__ void yolo_rgb_preprocess(
    GM_ADDR input, GM_ADDR output, int32_t input_height, int32_t input_width,
    int32_t output_height, int32_t output_width, int32_t resized_height,
    int32_t resized_width, int32_t pad_top, int32_t pad_left,
    int32_t source_height, int32_t source_width, int32_t pad_value,
    float scale, int32_t block_count)
{
    // Output pixels per tile. Staging a whole row would need the whole source
    // row in UB, which is fine for 640 but not for an arbitrarily wide input;
    // chunking keeps the staged span proportional to the chunk instead.
    constexpr int32_t kChunk = 128;

    AscendC::GlobalTensor<uint8_t> input_tensor;
    AscendC::GlobalTensor<float> output_tensor;
    AscendC::TPipe pipe;
    input_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(input),
                                 input_height * input_width * 3);
    const int32_t plane_size = output_height * output_width;
    const int32_t total_size = plane_size * 3;
    output_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output),
                                  total_size);

    // Two staged source rows, rounded up to whole 32 byte bursts because that is
    // the only shape DataCopy handles efficiently.
    const int32_t row_bytes = input_width * 3;
    const int32_t span_bytes = ((source_width * 3 + 31) / 32) * 32 + 32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> row_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> chunk_buf;
    pipe.InitBuffer(row_buf, span_bytes * 2);
    pipe.InitBuffer(chunk_buf, kChunk * sizeof(float));
    AscendC::LocalTensor<uint8_t> rows = row_buf.Get<uint8_t>();
    AscendC::LocalTensor<float> chunk = chunk_buf.Get<float>();

    const float pad = static_cast<float>(pad_value) * scale;
    const int32_t chunks_per_row = (output_width + kChunk - 1) / kChunk;
    const int32_t total_tiles = output_height * chunks_per_row;

    for (int32_t tile = AscendC::GetBlockIdx(); tile < total_tiles;
         tile += block_count) {
        const int32_t output_y = tile / chunks_per_row;
        const int32_t chunk_x = (tile - output_y * chunks_per_row) * kChunk;
        int32_t length = output_width - chunk_x;
        if (length > kChunk) {
            length = kChunk;
        }

        const int32_t resized_y = output_y - pad_top;
        const bool row_inside = resized_y >= 0 && resized_y < resized_height;
        int32_t y0 = 0;
        int32_t y1 = 0;
        float weight_y = 0.0f;
        if (row_inside) {
            MapY(output_y, pad_top, source_height, resized_height, &y0, &y1,
                 &weight_y);
        }

        // Stage exactly the source span this chunk reads. The mapping is
        // monotonic in x, so the span runs from the first to the last column the
        // chunk touches.
        int32_t span_start = 0;
        int32_t span_need = 0;
        int32_t bulk = 0;
        if (row_inside) {
            int32_t first_x = chunk_x > pad_left ? chunk_x : pad_left;
            int32_t last_x = chunk_x + length - 1;
            const int32_t inside_end = pad_left + resized_width - 1;
            if (last_x > inside_end) {
                last_x = inside_end;
            }
            if (last_x >= first_x) {
                int32_t x0 = 0;
                int32_t x1 = 0;
                float weight = 0.0f;
                MapX(first_x, pad_left, source_width, resized_width, &x0, &x1,
                     &weight);
                // Start on a burst boundary. A chunk never reads before its
                // first column, so backing up only fetches bytes nobody looks
                // at, and it keeps the source side of the copy aligned.
                span_start = (x0 * 3) & ~31;
                MapX(last_x, pad_left, source_width, resized_width, &x0, &x1,
                     &weight);
                span_need = (x1 + 1) * 3 - span_start;

                // DataCopy moves whole 32 byte bursts only, and a row is not
                // padded out to one, so the burst part stops at the last whole
                // burst inside the row and the remainder is fetched one byte at
                // a time below. A row is never over-read: source_width is not
                // always the image width, so the last row of the image would
                // otherwise leave the buffer.
                int32_t burst_end = (span_start + span_need) & ~31;
                if (burst_end > row_bytes) {
                    burst_end = row_bytes & ~31;
                }
                bulk = burst_end > span_start ? burst_end - span_start : 0;
            }
        }

        if (bulk > 0) {
            AscendC::DataCopy(rows, input_tensor[y0 * row_bytes + span_start],
                              bulk);
            AscendC::DataCopy(rows[span_bytes],
                              input_tensor[y1 * row_bytes + span_start], bulk);
        }
        for (int32_t byte = bulk; byte < span_need; ++byte) {
            rows.SetValue(
                byte, input_tensor.GetValue(y0 * row_bytes + span_start + byte));
            rows.SetValue(span_bytes + byte,
                          input_tensor.GetValue(y1 * row_bytes + span_start +
                                                byte));
        }

        for (int32_t plane = 0; plane < 3; ++plane) {
            // The image is decoded BGR, so plane 0 wants the red byte.
            const int32_t channel = 2 - plane;
            for (int32_t index = 0; index < length; ++index) {
                const int32_t output_x = chunk_x + index;
                const int32_t resized_x = output_x - pad_left;
                float value = pad;
                if (row_inside && resized_x >= 0 && resized_x < resized_width) {
                    int32_t x0 = 0;
                    int32_t x1 = 0;
                    float weight_x = 0.0f;
                    MapX(output_x, pad_left, source_width, resized_width, &x0,
                         &x1, &weight_x);
                    const int32_t offset0 = x0 * 3 + channel - span_start;
                    const int32_t offset1 = x1 * 3 + channel - span_start;
                    const float top =
                        static_cast<float>(ReadBgr(rows, offset0)) *
                            (1.0f - weight_x) +
                        static_cast<float>(ReadBgr(rows, offset1)) * weight_x;
                    const float bottom =
                        static_cast<float>(ReadBgr(rows, span_bytes + offset0)) *
                            (1.0f - weight_x) +
                        static_cast<float>(ReadBgr(rows, span_bytes + offset1)) *
                            weight_x;
                    value =
                        (top * (1.0f - weight_y) + bottom * weight_y) * scale;
                }
                chunk.SetValue(index, value);
            }

            // Both ends of the store have to line up with a 32 byte burst; the
            // tail that does not is written one element at a time.
            const int32_t base =
                plane * plane_size + output_y * output_width + chunk_x;
            const int32_t burst = (base & 7) == 0 ? (length & ~7) : 0;
            if (burst > 0) {
                AscendC::DataCopy(output_tensor[base], chunk, burst);
            }
            for (int32_t index = burst; index < length; ++index) {
                output_tensor.SetValue(base + index, chunk.GetValue(index));
            }
        }
    }
}

#ifndef ASCENDC_CPU_DEBUG
void yolo_rgb_preprocess_do(uint32_t block_dim, void* stream, uint8_t* input,
                            uint8_t* output, int32_t input_height,
                            int32_t input_width, int32_t output_height,
                            int32_t output_width, int32_t resized_height,
                            int32_t resized_width, int32_t pad_top,
                            int32_t pad_left, int32_t source_height,
                            int32_t source_width, int32_t pad_value,
                            float scale)
{
    yolo_rgb_preprocess<<<block_dim, nullptr, stream>>>(
        input, output, input_height, input_width, output_height, output_width,
        resized_height, resized_width, pad_top, pad_left, source_height,
        source_width, pad_value, scale, static_cast<int32_t>(block_dim));
}
#endif
