#ifndef WAVENET_LAYER_EIGEN_H_INCLUDED
#define WAVENET_LAYER_EIGEN_H_INCLUDED

#include "../Layer.h"
#include "../common.h"
#include "../config.h"
#include <Eigen/Dense>
#include <array>
#include <cmath>

namespace RTNEURAL_NAMESPACE
{

/**
 * Block size for block-based inference.
 * Processing samples in blocks reduces overhead and improves cache utilization.
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
    GatedTanh // tanh(x) * sigmoid(x)
};

/**
 * Static implementation of a fused WaveNet layer for NAM models.
 *
 * This layer fuses the following operations:
 * - Dilated causal convolution (1D, kernel_size typically 3)
 * - Bias addition
 * - Activation function
 * - Residual connection (input + activated output)
 * - Skip connection accumulation (optional)
 *
 * @tparam T Data type (float or double)
 * @tparam in_sizet Input/output channel size
 * @tparam kernel_size Convolution kernel size (typically 3 for WaveNet)
 * @tparam dilation_rate Dilation rate for the dilated convolution
 * @tparam activation Activation function to use
 * @tparam has_residual Whether to add residual connection
 * @tparam has_skip Whether to accumulate skip connection
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
    using vec_type = Eigen::Vector<T, in_sizet>;
    using weights_type = Eigen::Matrix<T, in_sizet, kernel_size>;
    using state_ptrs_type = Eigen::Vector<int, kernel_size>;

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = in_sizet;
    static constexpr auto block_size = WAVENET_BLOCK_SIZE;
    static constexpr auto state_size = (kernel_size - 1) * dilation_rate + 1;

    WaveNetLayerT()
        : outs(outs_internal)
        , skip_outs(skip_internal)
    {
        reset();
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "wavenet"; }

    /** Returns false since this is not a pure activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Resets the layer state. */
    RTNEURAL_REALTIME void reset()
    {
        state.setZero();
        state_ptr = 0;

        for (int i = 0; i < out_size; ++i)
        {
            outs_internal[i] = (T)0;
            skip_internal[i] = (T)0;
        }
    }

    /**
     * Performs forward propagation for a single sample.
     * This is the per-sample fallback path.
     */
    RTNEURAL_REALTIME inline void forward(const Eigen::Matrix<T, in_size, 1>& ins) noexcept
    {
        // Store input in circular buffer
        state.col(state_ptr) = ins;

        // Set state pointers for this sample
        setStatePointers();

        // Copy relevant columns for convolution
        for (int k = 0; k < kernel_size; ++k)
            state_cols.col(k) = state.col(state_ptrs(k));

        // Fused convolution + bias + activation + residual
        for (int i = 0; i < out_size; ++i)
        {
            T conv_out = state_cols.cwiseProduct(weights[i]).sum() + bias(i);

            T activated = applyActivation(conv_out);

            // Residual connection: output = input + activated
            if (has_residual)
                outs(i) = ins(i) + activated;
            else
                outs(i) = activated;

            // Skip connection accumulation
            if (has_skip)
                skip_outs(i) += activated;
        }

        // Advance state pointer
        state_ptr = (state_ptr == state_size - 1 ? 0 : state_ptr + 1);
    }

    /**
     * Performs block-based forward propagation.
     * Processes WAVENET_BLOCK_SIZE samples at once for improved performance.
     *
     * @param ins Input samples [in_size x block_size]
     * @param output Output samples [out_size x block_size]
     */
    RTNEURAL_REALTIME inline void forwardBlock(
        const T* ins,
        T* output) noexcept
    {
        for (int sample = 0; sample < block_size; ++sample)
        {
            // Map input sample
            Eigen::Map<const Eigen::Matrix<T, in_size, 1>> in_vec(ins + sample * in_size);

            // Store input in circular buffer
            state.col(state_ptr) = in_vec;

            // Set state pointers for this sample
            setStatePointers();

            // Copy relevant columns for convolution
            for (int k = 0; k < kernel_size; ++k)
                state_cols.col(k) = state.col(state_ptrs(k));

            // Fused convolution + bias + activation + residual
            for (int i = 0; i < out_size; ++i)
            {
                T conv_out = state_cols.cwiseProduct(weights[i]).sum() + bias(i);

                T activated = applyActivation(conv_out);

                // Store output
                if (has_residual)
                    output[sample * out_size + i] = in_vec(i) + activated;
                else
                    output[sample * out_size + i] = activated;

                // Skip connection accumulation
                if (has_skip)
                    skip_outs(i) += activated;
            }

            // Advance state pointer
            state_ptr = (state_ptr == state_size - 1 ? 0 : state_ptr + 1);
        }

        // Update outs with the last sample's output
        for (int i = 0; i < out_size; ++i)
            outs(i) = output[(block_size - 1) * out_size + i];
    }

    /**
     * Resets skip accumulator to zero.
     * Call this at the start of each block before processing.
     */
    RTNEURAL_REALTIME void resetSkip() noexcept
    {
        for (int i = 0; i < out_size; ++i)
            skip_internal[i] = (T)0;
    }

    /**
     * Gets the accumulated skip output.
     */
    RTNEURAL_REALTIME const T* getSkipOutput() const noexcept
    {
        return skip_internal;
    }

    /**
     * Sets the layer weights.
     *
     * The weights vector must have size weights[out_size][in_size][kernel_size]
     */
    RTNEURAL_REALTIME void setWeights(const std::vector<std::vector<std::vector<T>>>& w)
    {
        for (int i = 0; i < out_size; ++i)
        {
            for (int k = 0; k < kernel_size; ++k)
            {
                for (int j = 0; j < in_size; ++j)
                    weights[i](j, k) = w[i][j][k];
            }
        }
    }

    /**
     * Sets the layer biases.
     *
     * The bias vector must have size bias[out_size]
     */
    RTNEURAL_REALTIME void setBias(const std::vector<T>& biasVals)
    {
        for (int i = 0; i < out_size; ++i)
            bias(i) = biasVals[i];
    }

    /** Returns the size of the convolution kernel. */
    RTNEURAL_REALTIME int getKernelSize() const noexcept { return kernel_size; }

    /** Returns the convolution dilation rate. */
    RTNEURAL_REALTIME int getDilationRate() const noexcept { return dilation_rate; }

    Eigen::Map<vec_type, RTNeuralEigenAlignment> outs;
    Eigen::Map<vec_type, RTNeuralEigenAlignment> skip_outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
    T skip_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

    Eigen::Matrix<T, in_size, state_size> state;
    weights_type state_cols;

    int state_ptr = 0;
    state_ptrs_type state_ptrs;

    weights_type weights[out_size];
    vec_type bias;

    /** Apply the configured activation function. */
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
            // For gated activation, we expect the weights to produce 2x outputs
            // but here we simplify to tanh(x) * sigmoid(x)
            return std::tanh(x) * ((T)1 / ((T)1 + std::exp(-x)));
        default:
            return std::tanh(x);
        }
    }

    /** Sets pointers to state array columns. */
    RTNEURAL_REALTIME inline void setStatePointers()
    {
        for (int k = 0; k < kernel_size; ++k)
            state_ptrs[k] = (state_ptr + state_size - k * dilation_rate) % state_size;
    }
};

/**
 * Fused WaveNet block for NAM models.
 *
 * This class represents a complete WaveNet block that processes multiple
 * layers with increasing dilation rates. It provides both per-sample
 * and block-based inference.
 *
 * @tparam T Data type (float or double)
 * @tparam channel_size Number of channels
 * @tparam kernel_size Convolution kernel size
 * @tparam num_layers Number of layers in the block
 * @tparam max_dilation Maximum dilation rate (dilation = 2^layer_idx)
 */
template <typename T,
    int channel_size,
    int kernel_size,
    int num_layers,
    WaveNetActivation activation = WaveNetActivation::Tanh>
class WaveNetBlockT
{
    using vec_type = Eigen::Vector<T, channel_size>;

    // Calculate state sizes for each layer
    static constexpr int getDilationRate(int layer)
    {
        return 1 << layer; // 2^layer: 1, 2, 4, 8, 16, ...
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
        : outs(outs_internal)
    {
        reset();
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "wavenet_block"; }

    /** Returns false since this is not a pure activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Resets the layer state. */
    RTNEURAL_REALTIME void reset()
    {
        for (int layer = 0; layer < num_layers; ++layer)
        {
            states[layer].setZero();
            state_ptrs[layer] = 0;
        }

        skip_sum.setZero();

        for (int i = 0; i < out_size; ++i)
            outs_internal[i] = (T)0;
    }

    /**
     * Performs forward propagation for a single sample.
     */
    RTNEURAL_REALTIME inline void forward(const Eigen::Matrix<T, in_size, 1>& ins) noexcept
    {
        skip_sum.setZero();
        Eigen::Matrix<T, channel_size, 1> x = ins;

        for (int layer = 0; layer < num_layers; ++layer)
        {
            const int dilation = getDilationRate(layer);
            const int layer_state_size = getStateSize(dilation);

            // Store input in layer's circular buffer
            states[layer].col(state_ptrs[layer]) = x;

            // Compute convolution
            Eigen::Matrix<T, channel_size, 1> conv_out;
            conv_out.setZero();

            for (int k = 0; k < kernel_size; ++k)
            {
                int idx = (state_ptrs[layer] + layer_state_size - k * dilation) % layer_state_size;
                conv_out += weights[layer][k] * states[layer].col(idx);
            }

            conv_out += biases[layer];

            // Apply activation and residual
            for (int i = 0; i < channel_size; ++i)
            {
                T activated = applyActivation(conv_out(i));
                skip_sum(i) += activated;
                x(i) = x(i) + activated; // Residual connection
            }

            // Advance state pointer
            state_ptrs[layer] = (state_ptrs[layer] == layer_state_size - 1 ? 0 : state_ptrs[layer] + 1);
        }

        // Output is the skip sum (or last layer output depending on architecture)
        for (int i = 0; i < out_size; ++i)
            outs_internal[i] = skip_sum(i);
    }

    /**
     * Performs block-based forward propagation for improved performance.
     *
     * @param ins Input samples [in_size x block_size], stored as [sample0_ch0, sample0_ch1, ..., sample1_ch0, ...]
     * @param output Output samples [out_size x block_size]
     */
    RTNEURAL_REALTIME inline void forwardBlock(
        const T* ins,
        T* output) noexcept
    {
        for (int sample = 0; sample < block_size; ++sample)
        {
            // Map input sample
            Eigen::Map<const Eigen::Matrix<T, in_size, 1>> in_vec(ins + sample * in_size);

            forward(in_vec);

            // Copy output
            for (int i = 0; i < out_size; ++i)
                output[sample * out_size + i] = outs_internal[i];
        }
    }

    /**
     * Sets the layer weights for a specific layer.
     *
     * @param layer Layer index (0 to num_layers-1)
     * @param w Weights matrix [channel_size x channel_size] for each kernel position
     */
    RTNEURAL_REALTIME void setLayerWeights(int layer, int kernel_pos, const std::vector<std::vector<T>>& w)
    {
        if (layer >= num_layers || kernel_pos >= kernel_size)
            return;

        for (int i = 0; i < channel_size; ++i)
            for (int j = 0; j < channel_size; ++j)
                weights[layer][kernel_pos](i, j) = w[i][j];
    }

    /**
     * Sets the layer biases for a specific layer.
     */
    RTNEURAL_REALTIME void setLayerBias(int layer, const std::vector<T>& biasVals)
    {
        if (layer >= num_layers)
            return;

        for (int i = 0; i < channel_size; ++i)
            biases[layer](i) = biasVals[i];
    }

    Eigen::Map<vec_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

    // State buffers for each layer (max state size to accommodate all dilation rates)
    Eigen::Matrix<T, channel_size, max_state_size> states[num_layers];
    int state_ptrs[num_layers];

    // Weights: [layer][kernel_pos] is a channel_size x channel_size matrix
    Eigen::Matrix<T, channel_size, channel_size> weights[num_layers][kernel_size];
    Eigen::Vector<T, channel_size> biases[num_layers];

    // Skip connection accumulator
    Eigen::Vector<T, channel_size> skip_sum;

    /** Apply the configured activation function. */
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

#endif // WAVENET_LAYER_EIGEN_H_INCLUDED
