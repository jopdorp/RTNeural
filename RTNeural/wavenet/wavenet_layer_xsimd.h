#ifndef WAVENET_LAYER_XSIMD_H_INCLUDED
#define WAVENET_LAYER_XSIMD_H_INCLUDED

#include "../Layer.h"
#include "../common.h"
#include "../config.h"
#include <array>
#include <cmath>

namespace RTNEURAL_NAMESPACE
{

/**
 * Block size for block-based inference.
 */
static constexpr int WAVENET_BLOCK_SIZE = 32;

/**
 * Activation types supported by the WaveNet layer.
 */
enum class WaveNetActivation
{
    Tanh,
    Sigmoid,
    ReLU,
    GatedTanh
};

/**
 * Static implementation of a fused WaveNet layer for NAM models (XSIMD backend).
 */
template <typename T,
    int in_sizet,
    int kernel_size,
    int dilation_rate,
    WaveNetActivation activation = WaveNetActivation::Tanh,
    bool has_residual = true,
    bool has_skip = false>
class WaveNetLayerT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_io_size = ceil_div(in_sizet, v_size);

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = in_sizet;
    static constexpr auto block_size = WAVENET_BLOCK_SIZE;
    static constexpr auto state_size = (kernel_size - 1) * dilation_rate + 1;

    WaveNetLayerT()
    {
        reset();
    }

    std::string getName() const noexcept { return "wavenet"; }
    constexpr bool isActivation() const noexcept { return false; }

    RTNEURAL_REALTIME void reset()
    {
        for (int col = 0; col < state_size; ++col)
            for (int i = 0; i < v_io_size; ++i)
                state[col][i] = v_type((T)0);

        state_ptr = 0;

        for (int i = 0; i < v_io_size; ++i)
        {
            outs[i] = v_type((T)0);
            skip_outs[i] = v_type((T)0);
        }
    }

    RTNEURAL_REALTIME inline void forward(const v_type (&ins)[v_io_size]) noexcept
    {
        // Store input in circular buffer
        for (int i = 0; i < v_io_size; ++i)
            state[state_ptr][i] = ins[i];

        // Set state pointers
        setStatePointers();

        // Copy relevant columns for convolution
        for (int k = 0; k < kernel_size; ++k)
            for (int i = 0; i < v_io_size; ++i)
                state_cols[k][i] = state[state_ptrs[k]][i];

        // Fused convolution + bias + activation + residual
        for (int i = 0; i < v_io_size; ++i)
        {
            v_type conv_out = bias[i];

            for (int k = 0; k < kernel_size; ++k)
            {
                for (int j = 0; j < v_io_size; ++j)
                {
                    // Element-wise multiply and accumulate
                    conv_out += weights[i][k][j] * state_cols[k][j];
                }
            }

            v_type activated = applyActivation(conv_out);

            if (has_residual)
                outs[i] = ins[i] + activated;
            else
                outs[i] = activated;

            if (has_skip)
                skip_outs[i] += activated;
        }

        state_ptr = (state_ptr == state_size - 1 ? 0 : state_ptr + 1);
    }

    RTNEURAL_REALTIME inline void forwardBlock(const T* ins, T* output) noexcept
    {
        v_type sample_in[v_io_size];

        for (int sample = 0; sample < block_size; ++sample)
        {
            for (int i = 0; i < v_io_size; ++i)
                sample_in[i] = xsimd::load_aligned(ins + sample * in_size + i * v_size);

            forward(sample_in);

            for (int i = 0; i < v_io_size; ++i)
                xsimd::store_aligned(output + sample * out_size + i * v_size, outs[i]);
        }
    }

    RTNEURAL_REALTIME void resetSkip() noexcept
    {
        for (int i = 0; i < v_io_size; ++i)
            skip_outs[i] = v_type((T)0);
    }

    RTNEURAL_REALTIME void setWeights(const std::vector<std::vector<std::vector<T>>>& w)
    {
        for (int i = 0; i < out_size; ++i)
        {
            const int vi = i / v_size;
            const int vi_off = i % v_size;
            for (int j = 0; j < in_size; ++j)
            {
                const int vj = j / v_size;
                for (int k = 0; k < kernel_size; ++k)
                {
                    weights[vi][k][vj] = set_value(weights[vi][k][vj], vi_off, w[i][j][k]);
                }
            }
        }
    }

    RTNEURAL_REALTIME void setBias(const std::vector<T>& biasVals)
    {
        for (int i = 0; i < out_size; ++i)
        {
            const int vi = i / v_size;
            const int vi_off = i % v_size;
            bias[vi] = set_value(bias[vi], vi_off, biasVals[i]);
        }
    }

    RTNEURAL_REALTIME int getKernelSize() const noexcept { return kernel_size; }
    RTNEURAL_REALTIME int getDilationRate() const noexcept { return dilation_rate; }

    v_type outs[v_io_size];
    v_type skip_outs[v_io_size];

private:
    v_type state[state_size][v_io_size];
    v_type state_cols[kernel_size][v_io_size];
    int state_ptrs[kernel_size];
    int state_ptr = 0;

    v_type weights[v_io_size][kernel_size][v_io_size];
    v_type bias[v_io_size];

    RTNEURAL_REALTIME inline v_type applyActivation(v_type x) const noexcept
    {
        switch (activation)
        {
        case WaveNetActivation::Tanh:
            return xsimd::tanh(x);
        case WaveNetActivation::Sigmoid:
            return v_type((T)1) / (v_type((T)1) + xsimd::exp(-x));
        case WaveNetActivation::ReLU:
            return xsimd::max(x, v_type((T)0));
        case WaveNetActivation::GatedTanh:
            return xsimd::tanh(x) * (v_type((T)1) / (v_type((T)1) + xsimd::exp(-x)));
        default:
            return xsimd::tanh(x);
        }
    }

    RTNEURAL_REALTIME inline void setStatePointers()
    {
        for (int k = 0; k < kernel_size; ++k)
            state_ptrs[k] = (state_ptr + state_size - k * dilation_rate) % state_size;
    }
};

/**
 * Fused WaveNet block for NAM models (XSIMD backend).
 */
template <typename T,
    int channel_size,
    int kernel_size,
    int num_layers,
    WaveNetActivation activation = WaveNetActivation::Tanh>
class WaveNetBlockT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_channel_size = ceil_div(channel_size, v_size);

    static constexpr int getDilationRate(int layer)
    {
        return 1 << layer;
    }

    static constexpr int getStateSize(int dilation)
    {
        return (kernel_size - 1) * dilation + 1;
    }

    static constexpr int max_dilation = getDilationRate(num_layers - 1);
    static constexpr int max_state_size = getStateSize(max_dilation);

public:
    static constexpr auto in_size = channel_size;
    static constexpr auto out_size = channel_size;
    static constexpr auto block_size = WAVENET_BLOCK_SIZE;

    WaveNetBlockT()
    {
        reset();
    }

    std::string getName() const noexcept { return "wavenet_block"; }
    constexpr bool isActivation() const noexcept { return false; }

    RTNEURAL_REALTIME void reset()
    {
        for (int layer = 0; layer < num_layers; ++layer)
        {
            for (int col = 0; col < max_state_size; ++col)
                for (int i = 0; i < v_channel_size; ++i)
                    states[layer][col][i] = v_type((T)0);
            state_ptrs[layer] = 0;
        }

        for (int i = 0; i < v_channel_size; ++i)
        {
            skip_sum[i] = v_type((T)0);
            outs[i] = v_type((T)0);
        }
    }

    RTNEURAL_REALTIME inline void forward(const v_type (&ins)[v_channel_size]) noexcept
    {
        for (int i = 0; i < v_channel_size; ++i)
            skip_sum[i] = v_type((T)0);

        v_type x[v_channel_size];
        for (int i = 0; i < v_channel_size; ++i)
            x[i] = ins[i];

        for (int layer = 0; layer < num_layers; ++layer)
        {
            const int dilation = getDilationRate(layer);
            const int layer_state_size = getStateSize(dilation);

            // Store input in layer's circular buffer
            for (int i = 0; i < v_channel_size; ++i)
                states[layer][state_ptrs[layer]][i] = x[i];

            // Compute convolution
            v_type conv_out[v_channel_size];
            for (int i = 0; i < v_channel_size; ++i)
                conv_out[i] = biases[layer][i];

            for (int k = 0; k < kernel_size; ++k)
            {
                int idx = (state_ptrs[layer] + layer_state_size - k * dilation) % layer_state_size;

                for (int i = 0; i < v_channel_size; ++i)
                    for (int j = 0; j < v_channel_size; ++j)
                        conv_out[i] += weights[layer][k][i][j] * states[layer][idx][j];
            }

            // Apply activation and residual
            for (int i = 0; i < v_channel_size; ++i)
            {
                v_type activated = applyActivation(conv_out[i]);
                skip_sum[i] += activated;
                x[i] += activated;
            }

            // Advance state pointer
            state_ptrs[layer] = (state_ptrs[layer] == layer_state_size - 1 ? 0 : state_ptrs[layer] + 1);
        }

        for (int i = 0; i < v_channel_size; ++i)
            outs[i] = skip_sum[i];
    }

    RTNEURAL_REALTIME inline void forwardBlock(const T* ins, T* output) noexcept
    {
        v_type sample_in[v_channel_size];

        for (int sample = 0; sample < block_size; ++sample)
        {
            for (int i = 0; i < v_channel_size; ++i)
                sample_in[i] = xsimd::load_aligned(ins + sample * in_size + i * v_size);

            forward(sample_in);

            for (int i = 0; i < v_channel_size; ++i)
                xsimd::store_aligned(output + sample * out_size + i * v_size, outs[i]);
        }
    }

    RTNEURAL_REALTIME void setLayerWeights(int layer, int kernel_pos, const std::vector<std::vector<T>>& w)
    {
        if (layer >= num_layers || kernel_pos >= kernel_size)
            return;

        for (int i = 0; i < channel_size; ++i)
        {
            const int vi = i / v_size;
            const int vi_off = i % v_size;
            for (int j = 0; j < channel_size; ++j)
            {
                const int vj = j / v_size;
                weights[layer][kernel_pos][vi][vj] = set_value(weights[layer][kernel_pos][vi][vj], vi_off, w[i][j]);
            }
        }
    }

    RTNEURAL_REALTIME void setLayerBias(int layer, const std::vector<T>& biasVals)
    {
        if (layer >= num_layers)
            return;

        for (int i = 0; i < channel_size; ++i)
        {
            const int vi = i / v_size;
            const int vi_off = i % v_size;
            biases[layer][vi] = set_value(biases[layer][vi], vi_off, biasVals[i]);
        }
    }

    v_type outs[v_channel_size];

private:
    v_type states[num_layers][max_state_size][v_channel_size];
    int state_ptrs[num_layers];

    v_type weights[num_layers][kernel_size][v_channel_size][v_channel_size];
    v_type biases[num_layers][v_channel_size];

    v_type skip_sum[v_channel_size];

    RTNEURAL_REALTIME inline v_type applyActivation(v_type x) const noexcept
    {
        switch (activation)
        {
        case WaveNetActivation::Tanh:
            return xsimd::tanh(x);
        case WaveNetActivation::Sigmoid:
            return v_type((T)1) / (v_type((T)1) + xsimd::exp(-x));
        case WaveNetActivation::ReLU:
            return xsimd::max(x, v_type((T)0));
        case WaveNetActivation::GatedTanh:
            return xsimd::tanh(x) * (v_type((T)1) / (v_type((T)1) + xsimd::exp(-x)));
        default:
            return xsimd::tanh(x);
        }
    }
};

} // namespace RTNEURAL_NAMESPACE

#endif // WAVENET_LAYER_XSIMD_H_INCLUDED
