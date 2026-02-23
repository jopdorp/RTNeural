#ifndef WAVENET_LAYER_STL_H_INCLUDED
#define WAVENET_LAYER_STL_H_INCLUDED

#include "../Layer.h"
#include "../common.h"
#include "../config.h"
#include <array>
#include <cmath>
#include <vector>

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
 * Static implementation of a fused WaveNet layer for NAM models (STL backend).
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
            for (int i = 0; i < in_size; ++i)
                state[col][i] = (T)0;

        state_ptr = 0;

        for (int i = 0; i < out_size; ++i)
        {
            outs[i] = (T)0;
            skip_outs[i] = (T)0;
        }
    }

    RTNEURAL_REALTIME inline void forward(const T (&ins)[in_size]) noexcept
    {
        // Store input in circular buffer
        for (int i = 0; i < in_size; ++i)
            state[state_ptr][i] = ins[i];

        // Fused convolution + bias + activation + residual
        for (int i = 0; i < out_size; ++i)
        {
            T conv_out = bias[i];

            for (int k = 0; k < kernel_size; ++k)
            {
                // Branchless state index computation (no modulo in hot path)
                int idx = state_ptr - k * dilation_rate;
                idx += (idx < 0) * state_size;

                for (int j = 0; j < in_size; ++j)
                    conv_out += weights[i][k][j] * state[idx][j];
            }

            T activated = applyActivation(conv_out);

            if (has_residual)
                outs[i] = ins[i] + activated;
            else
                outs[i] = activated;

            if (has_skip)
                skip_outs[i] += activated;
        }

        // Advance state pointer (branchless wrap)
        ++state_ptr;
        state_ptr -= (state_ptr >= state_size) * state_size;
    }

    /**
     * True block-based forward propagation.
     * Processes block_size samples in a fused kernel.
     */
    RTNEURAL_REALTIME inline void forwardBlock(const T* ins, T* output) noexcept
    {
        // Precompute all state indices for the block
        int state_indices[block_size][kernel_size];
        int cur_ptr = state_ptr;
        for (int sample = 0; sample < block_size; ++sample)
        {
            for (int k = 0; k < kernel_size; ++k)
            {
                int idx = cur_ptr - k * dilation_rate;
                idx += (idx < 0) * state_size;
                state_indices[sample][k] = idx;
            }
            ++cur_ptr;
            cur_ptr -= (cur_ptr >= state_size) * state_size;
        }

        // Process all samples in the block
        cur_ptr = state_ptr;
        for (int sample = 0; sample < block_size; ++sample)
        {
            const T* in_ptr = ins + sample * in_size;
            T* out_ptr = output + sample * out_size;

            // Store input in circular buffer
            for (int i = 0; i < in_size; ++i)
                state[cur_ptr][i] = in_ptr[i];

            // Fused convolution + bias + activation + residual
            for (int i = 0; i < out_size; ++i)
            {
                T conv_out = bias[i];

                for (int k = 0; k < kernel_size; ++k)
                {
                    const int idx = state_indices[sample][k];
                    for (int j = 0; j < in_size; ++j)
                        conv_out += weights[i][k][j] * state[idx][j];
                }

                T activated = applyActivation(conv_out);

                if (has_residual)
                    out_ptr[i] = in_ptr[i] + activated;
                else
                    out_ptr[i] = activated;

                if (has_skip)
                    skip_outs[i] += activated;
            }

            // Advance state pointer (branchless wrap)
            ++cur_ptr;
            cur_ptr -= (cur_ptr >= state_size) * state_size;
        }

        // Update state pointer and outs from last sample
        state_ptr = cur_ptr;
        for (int i = 0; i < out_size; ++i)
            outs[i] = output[(block_size - 1) * out_size + i];
    }

    RTNEURAL_REALTIME void resetSkip() noexcept
    {
        for (int i = 0; i < out_size; ++i)
            skip_outs[i] = (T)0;
    }

    RTNEURAL_REALTIME const T* getSkipOutput() const noexcept
    {
        return skip_outs;
    }

    RTNEURAL_REALTIME void setWeights(const std::vector<std::vector<std::vector<T>>>& w)
    {
        for (int i = 0; i < out_size; ++i)
            for (int j = 0; j < in_size; ++j)
                for (int k = 0; k < kernel_size; ++k)
                    weights[i][k][j] = w[i][j][k];
    }

    RTNEURAL_REALTIME void setBias(const std::vector<T>& biasVals)
    {
        for (int i = 0; i < out_size; ++i)
            bias[i] = biasVals[i];
    }

    RTNEURAL_REALTIME int getKernelSize() const noexcept { return kernel_size; }
    RTNEURAL_REALTIME int getDilationRate() const noexcept { return dilation_rate; }

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
    T skip_outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

private:
    T state[state_size][in_size];
    int state_ptr = 0;

    T weights[out_size][kernel_size][in_size];
    T bias[out_size];

    RTNEURAL_REALTIME inline T applyActivation(T x) const noexcept
    {
        switch (activation)
        {
        case WaveNetActivation::Tanh:
            return std::tanh(x);
        case WaveNetActivation::Sigmoid:
            return (T)1 / ((T)1 + std::exp(-x));
        case WaveNetActivation::ReLU:
            return x > (T)0 ? x : (T)0;
        case WaveNetActivation::GatedTanh:
            return std::tanh(x) * ((T)1 / ((T)1 + std::exp(-x)));
        default:
            return std::tanh(x);
        }
    }
};

/**
 * Fused WaveNet block for NAM models (STL backend).
 */
template <typename T,
    int channel_size,
    int kernel_size,
    int num_layers,
    WaveNetActivation activation = WaveNetActivation::Tanh>
class WaveNetBlockT
{
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
                for (int i = 0; i < channel_size; ++i)
                    states[layer][col][i] = (T)0;
            state_ptrs[layer] = 0;
        }

        for (int i = 0; i < channel_size; ++i)
        {
            skip_sum[i] = (T)0;
            outs[i] = (T)0;
        }
    }

    RTNEURAL_REALTIME inline void forward(const T (&ins)[in_size]) noexcept
    {
        for (int i = 0; i < channel_size; ++i)
            skip_sum[i] = (T)0;

        T x[channel_size];
        for (int i = 0; i < channel_size; ++i)
            x[i] = ins[i];

        for (int layer = 0; layer < num_layers; ++layer)
        {
            const int dilation = getDilationRate(layer);
            const int layer_state_size = getStateSize(dilation);
            const int cur_ptr = state_ptrs[layer];

            // Store input in layer's circular buffer
            for (int i = 0; i < channel_size; ++i)
                states[layer][cur_ptr][i] = x[i];

            // Compute convolution
            T conv_out[channel_size];
            for (int i = 0; i < channel_size; ++i)
                conv_out[i] = biases[layer][i];

            for (int k = 0; k < kernel_size; ++k)
            {
                // Branchless index computation (no modulo)
                int idx = cur_ptr - k * dilation;
                idx += (idx < 0) * layer_state_size;

                for (int i = 0; i < channel_size; ++i)
                    for (int j = 0; j < channel_size; ++j)
                        conv_out[i] += weights[layer][k][i][j] * states[layer][idx][j];
            }

            // Apply activation and residual
            for (int i = 0; i < channel_size; ++i)
            {
                T activated = applyActivation(conv_out[i]);
                skip_sum[i] += activated;
                x[i] += activated;
            }

            // Advance state pointer (branchless wrap)
            int new_ptr = cur_ptr + 1;
            new_ptr -= (new_ptr >= layer_state_size) * layer_state_size;
            state_ptrs[layer] = new_ptr;
        }

        for (int i = 0; i < out_size; ++i)
            outs[i] = skip_sum[i];
    }

    /**
     * True block-based forward propagation.
     * Processes block_size samples in a fused kernel without calling forward().
     */
    RTNEURAL_REALTIME inline void forwardBlock(const T* ins, T* output) noexcept
    {
        // Process each sample through all layers
        for (int sample = 0; sample < block_size; ++sample)
        {
            const T* in_ptr = ins + sample * in_size;
            T* out_ptr = output + sample * out_size;

            // Initialize skip sum for this sample
            T sample_skip[channel_size];
            for (int i = 0; i < channel_size; ++i)
                sample_skip[i] = (T)0;

            // x holds the current layer's input
            T x[channel_size];
            for (int i = 0; i < channel_size; ++i)
                x[i] = in_ptr[i];

            // Process all layers
            for (int layer = 0; layer < num_layers; ++layer)
            {
                const int dilation = getDilationRate(layer);
                const int layer_state_size = getStateSize(dilation);
                const int cur_ptr = state_ptrs[layer];

                // Store x in layer's circular buffer
                for (int i = 0; i < channel_size; ++i)
                    states[layer][cur_ptr][i] = x[i];

                // Compute convolution + bias + activation + residual
                T conv_out[channel_size];
                for (int i = 0; i < channel_size; ++i)
                    conv_out[i] = biases[layer][i];

                for (int k = 0; k < kernel_size; ++k)
                {
                    // Branchless index computation
                    int idx = cur_ptr - k * dilation;
                    idx += (idx < 0) * layer_state_size;

                    for (int i = 0; i < channel_size; ++i)
                        for (int j = 0; j < channel_size; ++j)
                            conv_out[i] += weights[layer][k][i][j] * states[layer][idx][j];
                }

                // Fused activation + residual + skip
                for (int i = 0; i < channel_size; ++i)
                {
                    T activated = applyActivation(conv_out[i]);
                    sample_skip[i] += activated;
                    x[i] += activated;
                }

                // Advance state pointer (branchless wrap)
                int new_ptr = cur_ptr + 1;
                new_ptr -= (new_ptr >= layer_state_size) * layer_state_size;
                state_ptrs[layer] = new_ptr;
            }

            // Write output (skip sum)
            for (int i = 0; i < channel_size; ++i)
                out_ptr[i] = sample_skip[i];
        }

        // Update outs and skip_sum from last sample
        for (int i = 0; i < out_size; ++i)
        {
            outs[i] = output[(block_size - 1) * out_size + i];
            skip_sum[i] = outs[i];
        }
    }

    RTNEURAL_REALTIME void setLayerWeights(int layer, int kernel_pos, const std::vector<std::vector<T>>& w)
    {
        if (layer >= num_layers || kernel_pos >= kernel_size)
            return;

        for (int i = 0; i < channel_size; ++i)
            for (int j = 0; j < channel_size; ++j)
                weights[layer][kernel_pos][i][j] = w[i][j];
    }

    RTNEURAL_REALTIME void setLayerBias(int layer, const std::vector<T>& biasVals)
    {
        if (layer >= num_layers)
            return;

        for (int i = 0; i < channel_size; ++i)
            biases[layer][i] = biasVals[i];
    }

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

private:
    T states[num_layers][max_state_size][channel_size];
    int state_ptrs[num_layers];

    T weights[num_layers][kernel_size][channel_size][channel_size];
    T biases[num_layers][channel_size];

    T skip_sum[channel_size];

    RTNEURAL_REALTIME inline T applyActivation(T x) const noexcept
    {
        switch (activation)
        {
        case WaveNetActivation::Tanh:
            return std::tanh(x);
        case WaveNetActivation::Sigmoid:
            return (T)1 / ((T)1 + std::exp(-x));
        case WaveNetActivation::ReLU:
            return x > (T)0 ? x : (T)0;
        case WaveNetActivation::GatedTanh:
            return std::tanh(x) * ((T)1 / ((T)1 + std::exp(-x)));
        default:
            return std::tanh(x);
        }
    }
};

} // namespace RTNEURAL_NAMESPACE

#endif // WAVENET_LAYER_STL_H_INCLUDED
