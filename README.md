# Nano-GPT
We break the [video tutorial by Andrej Karpathy](https://youtu.be/kCc8FmEb1nY?si=5B0xvNJ2YYKVRX1u) on building a Generative Pre-trained Transformer (GPT) from scratch into 10 sections. The tutorial follows ideas covered in [Attention is All You Need](https://arxiv.org/abs/1706.03762)

Using PyTorch, we begin by coding a single layer NN bigram model that takes as input a single token and generates the next token. Then, we progress in complexity all the way to a decoder having 6 Transformer layers. The final model can take a sequence as input and generates the next token.


## Key concepts:
1. **Self-Attention Mechanism**:
   - The core innovation of the Transformer is the self-attention mechanism. It allows the model to weigh the importance of different parts of the input sequence when processing a particular token. This mechanism replaces the need for fixed-length context windows or recurrence. Because we construct a decoder, only previous parts of the sequence (before the current token) are considered by the self-attention mechanism.

   - **Scaled Dot-Product Attention**: Self-attention calculates attention scores by taking the dot product of a query vector with key vectors and scaling it. Then, it applies a softmax function to obtain attention weights. These weights are used to compute a weighted sum of the value vectors.

2. **Multi-Head Attention**:
   - Multi-head attention captures different types of relationships in the input data. Instead of using a single attention mechanism, multiple mechanisms (heads) are used in parallel.
   - Each head learns a different representation of the input sequence and contributes to the final output.

3. **Positional Encoding**:
   - Since the Transformer model does not have built-in positional information (unlike sequential models like RNNs), positional encodings are added to the input embeddings to give the model information about the position of tokens in the sequence.

4. **Feed-Forward Neural Networks**:
   - After the self-attention mechanism, each sub-layer in the decoder contains a feed-forward neural network.

5. **Residual Connections and Layer Normalization**:
   - To facilitate training deep networks, residual connections are used, and layer normalization is applied after each sub-layer.



