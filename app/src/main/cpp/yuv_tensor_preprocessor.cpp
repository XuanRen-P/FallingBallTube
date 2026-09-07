#include <jni.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

struct Affine {
    float xkx;
    float xky;
    float x_offset;
    float ykx;
    float yky;
    float y_offset;
};

Affine make_affine(
    const int rotation,
    const int crop_left,
    const int crop_top,
    const int crop_width,
    const int crop_height
) {
    switch (rotation) {
        case 90:
            return {
                0.0F,
                1.0F,
                static_cast<float>(crop_left),
                -1.0F,
                0.0F,
                static_cast<float>(crop_top + crop_height - 1)
            };
        case 180:
            return {
                -1.0F,
                0.0F,
                static_cast<float>(crop_left + crop_width - 1),
                0.0F,
                -1.0F,
                static_cast<float>(crop_top + crop_height - 1)
            };
        case 270:
            return {
                0.0F,
                -1.0F,
                static_cast<float>(crop_left + crop_width - 1),
                1.0F,
                0.0F,
                static_cast<float>(crop_top)
            };
        default:
            return {
                1.0F,
                0.0F,
                static_cast<float>(crop_left),
                0.0F,
                1.0F,
                static_cast<float>(crop_top)
            };
    }
}

int clamp_int(const int value, const int low, const int high) {
    return std::min(std::max(value, low), high);
}

float clamp_float(const float value, const float low, const float high) {
    return std::min(std::max(value, low), high);
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_cupet_fallingball_vision_YuvTensorPreprocessor_nativePreprocess(
    JNIEnv* env,
    jobject /* this_object */,
    jobject y_buffer,
    jobject u_buffer,
    jobject v_buffer,
    jobject output_buffer,
    jint frame_width,
    jint frame_height,
    jint y_row_stride,
    jint y_pixel_stride,
    jint u_row_stride,
    jint u_pixel_stride,
    jint v_row_stride,
    jint v_pixel_stride,
    jint rotation_degrees,
    jint crop_left,
    jint crop_top,
    jint crop_width,
    jint crop_height,
    jint roi_x,
    jint roi_y,
    jint roi_width,
    jint roi_height,
    jint model_size
) {
    const auto* y = static_cast<const std::uint8_t*>(env->GetDirectBufferAddress(y_buffer));
    const auto* u = static_cast<const std::uint8_t*>(env->GetDirectBufferAddress(u_buffer));
    const auto* v = static_cast<const std::uint8_t*>(env->GetDirectBufferAddress(v_buffer));
    auto* output = static_cast<float*>(env->GetDirectBufferAddress(output_buffer));
    const jlong y_capacity = env->GetDirectBufferCapacity(y_buffer);
    const jlong u_capacity = env->GetDirectBufferCapacity(u_buffer);
    const jlong v_capacity = env->GetDirectBufferCapacity(v_buffer);
    const jlong output_capacity = env->GetDirectBufferCapacity(output_buffer);

    const jlong channel_size = static_cast<jlong>(model_size) * model_size;
    const jlong required_output = 3 * channel_size;
    const bool dimensions_valid =
        frame_width > 0 && frame_height > 0 && crop_width > 0 && crop_height > 0 &&
        roi_width > 0 && roi_height > 0 && model_size > 0;
    const bool rotation_valid =
        rotation_degrees == 0 || rotation_degrees == 90 ||
        rotation_degrees == 180 || rotation_degrees == 270;
    if (y == nullptr || u == nullptr || v == nullptr || output == nullptr ||
        y_capacity <= 0 || u_capacity <= 0 || v_capacity <= 0 ||
        output_capacity < required_output || !dimensions_valid || !rotation_valid) {
        return JNI_FALSE;
    }

    const float model_scale =
        static_cast<float>(model_size) / static_cast<float>(std::max(roi_width, roi_height));
    const int new_width = std::max(
        1,
        static_cast<int>(std::nearbyint(static_cast<float>(roi_width) * model_scale))
    );
    const int new_height = std::max(
        1,
        static_cast<int>(std::nearbyint(static_cast<float>(roi_height) * model_scale))
    );
    const int pad_left = std::max(
        0,
        static_cast<int>(std::nearbyint(
            static_cast<float>(model_size - new_width) / 2.0F - 0.1F
        ))
    );
    const int pad_top = std::max(
        0,
        static_cast<int>(std::nearbyint(
            static_cast<float>(model_size - new_height) / 2.0F - 0.1F
        ))
    );
    constexpr float pad_value = 114.0F / 255.0F;
    std::fill(output, output + required_output, pad_value);

    const Affine affine = make_affine(
        rotation_degrees,
        crop_left,
        crop_top,
        crop_width,
        crop_height
    );
    const int max_chroma_x = (frame_width + 1) / 2 - 1;
    const int max_chroma_y = (frame_height + 1) / 2 - 1;
    const float source_scale_x = static_cast<float>(roi_width) / static_cast<float>(new_width);
    const float source_scale_y = static_cast<float>(roi_height) / static_cast<float>(new_height);
    constexpr float inverse_255 = 1.0F / 255.0F;

    const auto y_at = [&](const int x_value, const int y_value) -> float {
        const int x_clamped = clamp_int(x_value, 0, frame_width - 1);
        const int y_clamped = clamp_int(y_value, 0, frame_height - 1);
        const jlong index =
            static_cast<jlong>(y_clamped) * y_row_stride +
            static_cast<jlong>(x_clamped) * y_pixel_stride;
        return index >= 0 && index < y_capacity ? static_cast<float>(y[index]) : 0.0F;
    };
    const auto uv_at = [&](
        const int x_value,
        const int y_value,
        const std::uint8_t* plane,
        const int row_stride,
        const int pixel_stride,
        const jlong capacity
    ) -> int {
        const int x_clamped = clamp_int(x_value / 2, 0, max_chroma_x);
        const int y_clamped = clamp_int(y_value / 2, 0, max_chroma_y);
        const jlong index =
            static_cast<jlong>(y_clamped) * row_stride +
            static_cast<jlong>(x_clamped) * pixel_stride;
        return index >= 0 && index < capacity ? static_cast<int>(plane[index]) : 128;
    };

    for (int destination_y = 0; destination_y < new_height; ++destination_y) {
        const float logical_y =
            static_cast<float>(roi_y) +
            (static_cast<float>(destination_y) + 0.5F) * source_scale_y - 0.5F;
        const float row_buffer_x = affine.xky * logical_y + affine.x_offset;
        const float row_buffer_y = affine.yky * logical_y + affine.y_offset;
        const int destination_row =
            (pad_top + destination_y) * model_size + pad_left;

        for (int destination_x = 0; destination_x < new_width; ++destination_x) {
            const float logical_x =
                static_cast<float>(roi_x) +
                (static_cast<float>(destination_x) + 0.5F) * source_scale_x - 0.5F;
            const float buffer_x = affine.xkx * logical_x + row_buffer_x;
            const float buffer_y = affine.ykx * logical_x + row_buffer_y;
            const int x0 = static_cast<int>(std::floor(buffer_x));
            const int y0 = static_cast<int>(std::floor(buffer_y));
            const float fx = clamp_float(buffer_x - static_cast<float>(x0), 0.0F, 1.0F);
            const float fy = clamp_float(buffer_y - static_cast<float>(y0), 0.0F, 1.0F);
            const float y00 = y_at(x0, y0);
            const float y10 = y_at(x0 + 1, y0);
            const float y01 = y_at(x0, y0 + 1);
            const float y11 = y_at(x0 + 1, y0 + 1);
            const float y_upper = y00 + (y10 - y00) * fx;
            const float y_lower = y01 + (y11 - y01) * fx;
            const float y_value = y_upper + (y_lower - y_upper) * fy;
            const int chroma_x = static_cast<int>(buffer_x + 0.5F);
            const int chroma_y = static_cast<int>(buffer_y + 0.5F);
            const float u_value = static_cast<float>(
                uv_at(
                    chroma_x,
                    chroma_y,
                    u,
                    u_row_stride,
                    u_pixel_stride,
                    u_capacity
                ) - 128
            );
            const float v_value = static_cast<float>(
                uv_at(
                    chroma_x,
                    chroma_y,
                    v,
                    v_row_stride,
                    v_pixel_stride,
                    v_capacity
                ) - 128
            );
            const int destination = destination_row + destination_x;
            output[destination] = clamp_float(
                (y_value + 1.402F * v_value) * inverse_255,
                0.0F,
                1.0F
            );
            output[channel_size + destination] = clamp_float(
                (y_value - 0.34414F * u_value - 0.71414F * v_value) * inverse_255,
                0.0F,
                1.0F
            );
            output[2 * channel_size + destination] = clamp_float(
                (y_value + 1.772F * u_value) * inverse_255,
                0.0F,
                1.0F
            );
        }
    }

    return JNI_TRUE;
}
