Word Analogies with Vector Arithmetic & Virtual Threads (Java)

This application creates word analogies by applying vector arithmetic to word embeddings. A user supplies an embeddings file, an expression (e.g., **king - man + woman**), selects a metric, chooses a mode, and specifies topN results to output.

**Main features**

*   **Menu-driven CLI** (**Runner**) to input:

1.  embeddings file path, 2) expression, 3) output file path, plus options for metric, mode (in-memory/stream), and topN (_Runner.java)._

*   **Expression parsing & evaluation** (_ExpressionParser_) using shunting-yard + RPN with:

*   operators + - \* /, parentheses, unary minus
*   **scalar broadcasting** to vectors of length D (_ExpressionParser.java)._

*   **Vector similarity search** (_SearchEngine_):

*   COSINE, DOT, and EUCLIDEAN (implemented as negative squared distance so higher is better)
*   returns the top-N highest scoring tokens (_SearchEngine.java_).

*   **Parallel execution with virtual threads** (_ThreadManager_):

*   in-memory mode partitions embeddings and scores in parallel using Executors.newVirtualThreadPerTaskExecutor() (_ThreadManager.java)._

*   **Two modes:**

*   In-memory: loads all embeddings and performs parallel scoring.
*   Streaming: scans the embeddings file and loads only the required tokens.

**Additional features**

*   **Streaming mode** loads only required tokens by extracting word tokens from the expression ExpressionParser.java and scanning embeddings line-by-line (_EmbeddingLoader.java_).
*   **Robust embeddings loading**: supports comma-separated files (commas normalized to whitespace), skips malformed lines and dimension-mismatched vectors, and auto-detects embedding dimension for streaming (_EmbeddingLoader.java)_.
*   **UTF-8 output** (**IOManager**) writes rank, token, score, creating output directories if needed; tokens are sanitized to keep output one-line per result (_IOManager.java_).
*   **Emoji-enhanced CLI UI**: emojis added to menu and progress bar; correct UTF‑8 console output via ANSI reset handling. \* Run chcp 65001 \*

·         **Searching with emojis:** users can include emoji tokens directly in the expression (e.g.,  - man + woman). Results may also contain emoji tokens if they exist as tokens in the embeddings file.
