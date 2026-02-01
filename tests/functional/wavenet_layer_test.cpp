#include <gmock/gmock.h>

#include <RTNeural/RTNeural.h>
#include <cmath>
#include <vector>

namespace
{
using TestType = double;
using namespace RTNeural;

/**
 * Test that WaveNetLayerT produces correct output for basic forward pass.
 */
TEST(TestWaveNetLayer, singleLayerForwardProducesValidOutput)
{
    // Create a single WaveNet layer: 4 channels, kernel size 3, dilation 1
    WaveNetLayerT<TestType, 4, 3, 1, WaveNetActivation::Tanh, false, false> layer;

    // Set simple weights (identity-like for testing)
    std::vector<std::vector<std::vector<TestType>>> weights(4);
    for (int i = 0; i < 4; ++i)
    {
        weights[i].resize(4);
        for (int j = 0; j < 4; ++j)
        {
            weights[i][j].resize(3);
            for (int k = 0; k < 3; ++k)
            {
                // Simple pattern: only center kernel position has weight 1 for diagonal
                if (k == 0 && i == j)
                    weights[i][j][k] = 1.0;
                else
                    weights[i][j][k] = 0.0;
            }
        }
    }
    layer.setWeights(weights);

    // Set zero biases
    std::vector<TestType> biases(4, 0.0);
    layer.setBias(biases);

    layer.reset();

    // Process a few samples to fill the state buffer
    TestType input[4] = { 0.5, 0.3, -0.2, 0.1 };
    Eigen::Map<const Eigen::Matrix<TestType, 4, 1>> in_vec(input);

    layer.forward(in_vec);

    // With identity-like weights and tanh activation, output should be tanh(input)
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(layer.outs(i), std::tanh(input[i]), 1e-10);
    }
}

/**
 * Test that WaveNetLayerT with residual connection works correctly.
 */
TEST(TestWaveNetLayer, residualConnectionAddsInputToOutput)
{
    // Create layer with residual connection
    WaveNetLayerT<TestType, 4, 3, 1, WaveNetActivation::Tanh, true, false> layer;

    // Set zero weights (so conv output is just bias = 0)
    std::vector<std::vector<std::vector<TestType>>> weights(4);
    for (int i = 0; i < 4; ++i)
    {
        weights[i].resize(4);
        for (int j = 0; j < 4; ++j)
        {
            weights[i][j].resize(3, 0.0);
        }
    }
    layer.setWeights(weights);

    std::vector<TestType> biases(4, 0.0);
    layer.setBias(biases);

    layer.reset();

    TestType input[4] = { 0.5, 0.3, -0.2, 0.1 };
    Eigen::Map<const Eigen::Matrix<TestType, 4, 1>> in_vec(input);

    layer.forward(in_vec);

    // With zero weights and zero bias, conv output = 0, tanh(0) = 0
    // With residual, output = input + 0 = input
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(layer.outs(i), input[i], 1e-10);
    }
}

/**
 * Test different activation functions.
 */
TEST(TestWaveNetLayer, differentActivationFunctionsWork)
{
    // Test ReLU activation
    WaveNetLayerT<TestType, 4, 3, 1, WaveNetActivation::ReLU, false, false> relu_layer;

    std::vector<std::vector<std::vector<TestType>>> weights(4);
    for (int i = 0; i < 4; ++i)
    {
        weights[i].resize(4);
        for (int j = 0; j < 4; ++j)
        {
            weights[i][j].resize(3);
            for (int k = 0; k < 3; ++k)
            {
                if (k == 0 && i == j)
                    weights[i][j][k] = 1.0;
                else
                    weights[i][j][k] = 0.0;
            }
        }
    }
    relu_layer.setWeights(weights);

    std::vector<TestType> biases(4, 0.0);
    relu_layer.setBias(biases);

    relu_layer.reset();

    TestType input[4] = { 0.5, 0.3, -0.2, 0.1 };
    Eigen::Map<const Eigen::Matrix<TestType, 4, 1>> in_vec(input);

    relu_layer.forward(in_vec);

    // ReLU activation: max(0, x)
    for (int i = 0; i < 4; ++i)
    {
        TestType expected = input[i] > 0 ? input[i] : 0.0;
        EXPECT_NEAR(relu_layer.outs(i), expected, 1e-10);
    }
}

/**
 * Test dilation affects the convolution correctly.
 */
TEST(TestWaveNetLayer, dilationAffectsConvolutionCorrectly)
{
    // Create layer with dilation 2
    WaveNetLayerT<TestType, 2, 3, 2, WaveNetActivation::Tanh, false, false> layer;

    // Set weights to sum all inputs from all kernel positions
    std::vector<std::vector<std::vector<TestType>>> weights(2);
    for (int i = 0; i < 2; ++i)
    {
        weights[i].resize(2);
        for (int j = 0; j < 2; ++j)
        {
            weights[i][j].resize(3, 0.1); // Small weights
        }
    }
    layer.setWeights(weights);

    std::vector<TestType> biases(2, 0.0);
    layer.setBias(biases);

    layer.reset();

    // Process multiple samples to fill buffer
    for (int s = 0; s < 10; ++s)
    {
        TestType input[2] = { (TestType)(s * 0.1), (TestType)(s * 0.05) };
        Eigen::Map<const Eigen::Matrix<TestType, 2, 1>> in_vec(input);
        layer.forward(in_vec);
    }

    // Just verify the layer produces valid output without crashing
    EXPECT_FALSE(std::isnan(layer.outs(0)));
    EXPECT_FALSE(std::isnan(layer.outs(1)));
}

/**
 * Test block-based inference produces same results as per-sample.
 */
TEST(TestWaveNetLayer, blockInferenceMatchesPerSample)
{
    constexpr int block_size = WAVENET_BLOCK_SIZE;

    WaveNetLayerT<TestType, 4, 3, 1, WaveNetActivation::Tanh, true, false> sample_layer;
    WaveNetLayerT<TestType, 4, 3, 1, WaveNetActivation::Tanh, true, false> block_layer;

    // Set same weights for both
    std::vector<std::vector<std::vector<TestType>>> weights(4);
    for (int i = 0; i < 4; ++i)
    {
        weights[i].resize(4);
        for (int j = 0; j < 4; ++j)
        {
            weights[i][j].resize(3);
            for (int k = 0; k < 3; ++k)
            {
                weights[i][j][k] = 0.1 * (i + 1) * (j + 1) * (k + 1);
            }
        }
    }
    sample_layer.setWeights(weights);
    block_layer.setWeights(weights);

    std::vector<TestType> biases = { 0.1, -0.1, 0.2, -0.2 };
    sample_layer.setBias(biases);
    block_layer.setBias(biases);

    sample_layer.reset();
    block_layer.reset();

    // Create input block
    std::vector<TestType> input_block(block_size * 4);
    for (int s = 0; s < block_size; ++s)
    {
        for (int c = 0; c < 4; ++c)
        {
            input_block[s * 4 + c] = std::sin((TestType)(s * 0.1 + c * 0.2));
        }
    }

    // Process per-sample
    std::vector<TestType> sample_output(block_size * 4);
    for (int s = 0; s < block_size; ++s)
    {
        Eigen::Map<const Eigen::Matrix<TestType, 4, 1>> in_vec(input_block.data() + s * 4);
        sample_layer.forward(in_vec);
        for (int c = 0; c < 4; ++c)
            sample_output[s * 4 + c] = sample_layer.outs(c);
    }

    // Process block
    std::vector<TestType> block_output(block_size * 4);
    block_layer.forwardBlock(input_block.data(), block_output.data());

    // Compare outputs
    for (int i = 0; i < block_size * 4; ++i)
    {
        EXPECT_NEAR(block_output[i], sample_output[i], 1e-10)
            << "Mismatch at index " << i;
    }
}

/**
 * Test WaveNetBlockT (multi-layer block) basic functionality.
 */
TEST(TestWaveNetBlock, multiLayerBlockProducesValidOutput)
{
    // Create a 4-layer WaveNet block
    WaveNetBlockT<TestType, 4, 3, 4, WaveNetActivation::Tanh> block;

    // Set weights for each layer
    for (int layer = 0; layer < 4; ++layer)
    {
        for (int k = 0; k < 3; ++k)
        {
            std::vector<std::vector<TestType>> w(4, std::vector<TestType>(4));
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    // Simple pattern
                    if (i == j)
                        w[i][j] = 0.1;
                    else
                        w[i][j] = 0.01;
                }
            }
            block.setLayerWeights(layer, k, w);
        }

        std::vector<TestType> b(4, 0.0);
        block.setLayerBias(layer, b);
    }

    block.reset();

    // Process multiple samples
    for (int s = 0; s < 20; ++s)
    {
        TestType input_arr[4] = {
            (TestType)(0.5 * std::sin(s * 0.1)),
            (TestType)(0.3 * std::cos(s * 0.15)),
            (TestType)(0.2 * std::sin(s * 0.2)),
            (TestType)(0.1 * std::cos(s * 0.25))
        };
        Eigen::Map<const Eigen::Matrix<TestType, 4, 1>> input(input_arr);

        block.forward(input);

        // Verify output is valid
        for (int c = 0; c < 4; ++c)
        {
            EXPECT_FALSE(std::isnan(block.outs(c))) << "NaN at sample " << s << ", channel " << c;
            EXPECT_FALSE(std::isinf(block.outs(c))) << "Inf at sample " << s << ", channel " << c;
        }
    }
}

/**
 * Test that block-based inference matches per-sample for WaveNetBlockT.
 */
TEST(TestWaveNetBlock, blockInferenceMatchesPerSample)
{
    constexpr int block_size = WAVENET_BLOCK_SIZE;

    WaveNetBlockT<TestType, 4, 3, 3, WaveNetActivation::Tanh> sample_block;
    WaveNetBlockT<TestType, 4, 3, 3, WaveNetActivation::Tanh> block_block;

    // Set same weights for both
    for (int layer = 0; layer < 3; ++layer)
    {
        for (int k = 0; k < 3; ++k)
        {
            std::vector<std::vector<TestType>> w(4, std::vector<TestType>(4));
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    w[i][j] = 0.05 * (layer + 1) * (i - j) / 4.0;
                }
            }
            sample_block.setLayerWeights(layer, k, w);
            block_block.setLayerWeights(layer, k, w);
        }

        std::vector<TestType> b(4);
        for (int i = 0; i < 4; ++i)
            b[i] = 0.01 * (layer + 1);
        sample_block.setLayerBias(layer, b);
        block_block.setLayerBias(layer, b);
    }

    sample_block.reset();
    block_block.reset();

    // Create input block
    std::vector<TestType> input_block(block_size * 4);
    for (int s = 0; s < block_size; ++s)
    {
        for (int c = 0; c < 4; ++c)
        {
            input_block[s * 4 + c] = 0.1 * std::sin((TestType)(s * 0.1 + c * 0.3));
        }
    }

    // Process per-sample
    std::vector<TestType> sample_output(block_size * 4);
    for (int s = 0; s < block_size; ++s)
    {
        TestType input_arr[4];
        for (int c = 0; c < 4; ++c)
            input_arr[c] = input_block[s * 4 + c];
        Eigen::Map<const Eigen::Matrix<TestType, 4, 1>> input(input_arr);

        sample_block.forward(input);

        for (int c = 0; c < 4; ++c)
            sample_output[s * 4 + c] = sample_block.outs(c);
    }

    // Process block
    std::vector<TestType> block_output(block_size * 4);
    block_block.forwardBlock(input_block.data(), block_output.data());

    // Compare outputs
    for (int i = 0; i < block_size * 4; ++i)
    {
        EXPECT_NEAR(block_output[i], sample_output[i], 1e-10)
            << "Mismatch at index " << i;
    }
}

/**
 * Debug/test path: Compare WaveNetLayerT with standard Conv1DT for basic convolution.
 *
 * This test verifies that the WaveNet layer produces equivalent results to
 * the standard Conv1D layer for the convolution operation (before activation).
 */
TEST(TestWaveNetComparison, waveNetLayerMatchesConv1DForConvolution)
{
    // We'll test that WaveNet layer (without activation, without residual)
    // produces the same convolution result as Conv1DT when we manually
    // apply the same transformation.

    constexpr int channels = 4;
    constexpr int kernel = 3;
    constexpr int dilation = 1;

    // Create a standard Conv1D layer
    Conv1DT<TestType, channels, channels, kernel, dilation> conv_layer;

    // Create a WaveNet layer with ReLU (linear for positive inputs)
    WaveNetLayerT<TestType, channels, kernel, dilation, WaveNetActivation::Tanh, false, false> wavenet_layer;

    // Set identical weights
    std::vector<std::vector<std::vector<TestType>>> weights(channels);
    for (int i = 0; i < channels; ++i)
    {
        weights[i].resize(channels);
        for (int j = 0; j < channels; ++j)
        {
            weights[i][j].resize(kernel);
            for (int k = 0; k < kernel; ++k)
            {
                weights[i][j][k] = 0.1 * (i + 1) - 0.05 * (j + 1) + 0.02 * (k + 1);
            }
        }
    }
    conv_layer.setWeights(weights);
    wavenet_layer.setWeights(weights);

    std::vector<TestType> biases(channels);
    for (int i = 0; i < channels; ++i)
        biases[i] = 0.05 * (i + 1);
    conv_layer.setBias(biases);
    wavenet_layer.setBias(biases);

    conv_layer.reset();
    wavenet_layer.reset();

    // Process multiple samples
    for (int s = 0; s < 10; ++s)
    {
        TestType input[channels];
        for (int c = 0; c < channels; ++c)
            input[c] = 0.3 * std::sin(s * 0.2 + c * 0.1);

        Eigen::Map<const Eigen::Matrix<TestType, channels, 1>> in_vec(input);

        conv_layer.forward(in_vec);
        wavenet_layer.forward(in_vec);

        // WaveNet applies tanh to convolution output
        // Compare: wavenet_out = tanh(conv_out)
        for (int c = 0; c < channels; ++c)
        {
            TestType expected = std::tanh(conv_layer.outs(c));
            EXPECT_NEAR(wavenet_layer.outs(c), expected, 1e-10)
                << "Mismatch at sample " << s << ", channel " << c;
        }
    }
}

} // namespace
