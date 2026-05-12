package ie.atu.sw;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/**
 * SearchEngine
 *
 * In-memory search:
 * - partitions the embedding map into chunks
 * - scores each chunk in parallel using virtual threads (via ThreadManager)
 * - keeps per-task top-N heaps and merges them
 *
 * Streaming search:
 * - reads vectors line-by-line
 * - maintains a single top-N min-heap
 *
 * Big-O (as stated in the brief):
 * - In-memory: O(V * D)
 * - Streaming: O(V * D)
 */
public final class SearchEngine {

    // O(1)
    private SearchEngine() { }

    /**
     * Supported metrics.
     */
    public enum Metric { COSINE, DOT, EUCLIDEAN }

    /**
     * Result container for (token, score).
     */
    public static class Result {
        public final String token;
        public final double score;

        /**
         * Constructs a result.
         *
         * Big-O: O(1)
         * Rationale: stores references/primitive values.
         *
         * @param token token associated with the score
         * @param score similarity/distance score
         */
        public Result(String token, double score) {
            // O(1)
            this.token = token;
            this.score = score;
        }

        // O(1)
        @Override
        public String toString() {
            // O(1) for formatting (depends on token length, but asymptotically constant here)
            return token + "\t" + score;
        }
    }

    /**
     * In-memory search using partitioned parallel scoring.
     *
     * Let:
     * - V = number of vectors in {@code embeddings}
     * - D = embedding dimension
     *
     * Big-O:
     * - Time: O(V * D) dominated by scoring
     * - Space: O(V) for embeddings + O(V) entry list
     *
     * @param embeddings map token -> embedding vector (fully loaded in memory)
     * @param precomputedMags precomputed magnitudes for cosine (may be null)
     * @param queryVector query vector
     * @param metric metric choice
     * @param topN number of results to return
     * @return list of top-N results in descending score order
     * @throws InterruptedException if the parallel execution is interrupted
     */
    public static List<Result> searchInMemory(
            Map<String, double[]> embeddings,
            Map<String, Double> precomputedMags,
            double[] queryVector,
            Metric metric,
            int topN
    ) throws InterruptedException {

        // O(1)
        Objects.requireNonNull(embeddings);

        // O(1)
        if (embeddings.isEmpty()) return Collections.emptyList();

        // O(V) to copy map entries into a list
        // Rationale: ThreadManager partitions by index in an ArrayList.
        List<Map.Entry<String, double[]>> entries = new ArrayList<>(embeddings.entrySet());

        // O(1)
        Config cfg = new Config(topN, metric, 2, true);

        // ThreadManager.parallelTopN: dominated by scoring O(V * D)
        // Rationale: each vector score costs O(D) and we score V vectors.
        return ThreadManager.parallelTopN(entries, precomputedMags, queryVector, cfg);
    }

    /**
     * Streaming search: scans the embedding file once and maintains a top-N heap.
     *
     * Let:
     * - V = number of vectors/lines in the file
     * - D = embedding dimension
     *
     * Big-O:
     * - Time: O(V * D) dominated by scoring each vector
     * - Space: O(topN) for the heap plus O(1) variables
     *
     * @param embeddingFile path to the embeddings file
     * @param queryVector query vector
     * @param metric metric choice
     * @param topN number of results to return
     * @return list of top-N results in descending score order
     * @throws IOException if reading embeddings fails
     */
    public static List<Result> searchStreaming(
            Path embeddingFile,
            double[] queryVector,
            Metric metric,
            int topN
    ) throws IOException {

        // O(1)
        PriorityQueue<Result> heap =
                new PriorityQueue<>(Comparator.comparingDouble(r -> r.score));

        // O(D) only for cosine (magnitude calculation); else O(1)
        final double queryMag =
                (metric == Metric.COSINE) ? Vector.magnitude(queryVector) : 0.0;

        // O(V * D) overall:
        // - EmbeddingLoader.stream scans V lines
        // - For each vector, computeScore costs O(D)
        EmbeddingLoader.stream(embeddingFile, (token, vec) -> {
            // O(D) per computeScore call for dot/distance
            double score = computeScore(vec, null, queryVector, queryMag, metric);

            // O(log topN) heap update per considered vector
            consider(heap, new Result(token, score), topN);
        });

        // O(topN) to empty heap into a list
        List<Result> out = new ArrayList<>();
        while (!heap.isEmpty()) out.add(heap.poll());

        // O(topN) reverse into descending order
        Collections.reverse(out);

        // O(1) return reference
        return out;
    }

    /**
     * Maintains a min-heap of size up to topN.
     *
     * Big-O:
     * - O(log topN) for add/poll operations
     *
     * @param heap min-heap containing the best results seen so far
     * @param r candidate result
     * @param topN maximum heap size
     */
    private static void consider(PriorityQueue<Result> heap, Result r, int topN) {
        // O(log topN) for add / poll+add
        if (heap.size() < topN) heap.add(r);
        else if (heap.peek().score < r.score) {
            heap.poll();
            heap.add(r);
        }
    }

    /**
     * Computes similarity/distance score for a single vector.
     *
     * Big-O:
     * - DOT: O(D)
     * - COSINE: O(D) due to dot product (+ magnitude if not precomputed)
     * - EUCLIDEAN: O(D) for squared distance
     *
     * @param vec embedding vector to score against query
     * @param vecMag pre-computed magnitude of vec (may be null)
     * @param queryVector query vector
     * @param queryMag pre-computed magnitude of query vector (only used for cosine)
     * @param metric metric choice
     * @return score where higher means “more similar” (EUCLIDEAN returns negated squared distance)
     */
    private static double computeScore(
            double[] vec,
            Double vecMag,
            double[] queryVector,
            double queryMag,
            Metric metric
    ) {
        // Dominant computation is dot/distance: O(D)
        return switch (metric) {
            case DOT ->
                    Vector.dot(queryVector, vec); // O(D)

            case COSINE -> {
                // If vecMag is null, magnitude(vec) costs O(D); otherwise lookup is O(1)
                double magVec = (vecMag == null)
                        ? Vector.magnitude(vec) // O(D)
                        : vecMag;               // O(1)
                yield Vector.cosine(queryVector, vec, queryMag, magVec); // O(D)
            }

            case EUCLIDEAN ->
                    // Negated squared Euclidean so that "higher score is better"
                    -Vector.euclideanSquared(queryVector, vec); // O(D)
        };
    }
}