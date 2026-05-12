package ie.atu.sw;

import java.util.*;
import java.util.concurrent.*;

/**
 * ThreadManager
 *
 * Manages parallel search using virtual threads (or a fixed pool if disabled).
 */
public final class ThreadManager {

    // O(1)
    private ThreadManager() { }

    /**
     * Computes top-N results in parallel over an in-memory embeddings list.
     *
     * Let:
     * - V = number of vectors (entries.size())
     * - D = embedding dimension (vector length)
     * - parts = number of partitions/tasks
     * - n = topN
     *
     * Big-O (asymptotic):
     * - Time: O(V * D) dominated by scoring
     *         + O(V * log n) for heap maintenance (subdominant vs scoring when D >> log n)
     *         + O(parts * n * log n) heap merge overhead
     * - Space: O(parts * n) for per-task heaps (plus O(V) for the entries list held by caller)
     *
     * Rationale:
     * - Partitioning reduces wall-clock time without changing total scoring work.
     *
     * @param entries list of (token, vector) entries to score
     * @param precomputedMags precomputed magnitudes for cosine (may be null)
     * @param queryVector the query vector
     * @param cfg configuration options including metric and whether to use virtual threads
     * @return list of top-N results in descending score order
     * @throws InterruptedException if task execution is interrupted
     */
    public static List<SearchEngine.Result> parallelTopN(
            List<Map.Entry<String, double[]>> entries,
            Map<String, Double> precomputedMags,
            double[] queryVector,
            Config cfg
    ) throws InterruptedException {

        // O(1)
        Objects.requireNonNull(entries);
        Objects.requireNonNull(queryVector);
        Objects.requireNonNull(cfg);

        // O(1)
        if (entries.isEmpty()) return Collections.emptyList();

        // O(1)
        if (cfg.topN() <= 0) {
            throw new IllegalArgumentException("topN must be > 0");
        }

        // O(1)
        int procs = Math.max(1, Runtime.getRuntime().availableProcessors());

        // O(1)
        int partitionFactor = Math.max(1, cfg.partitionFactor());

        // O(1)
        int total = entries.size(); // V

        // O(1)
        int parts = Math.min(total, procs * partitionFactor);

        // O(1)
        int chunkSize = (total + parts - 1) / parts;

        // O(D) only when cosine is selected (magnitude of queryVector)
        final double queryMag =
                (cfg.metric() == SearchEngine.Metric.COSINE) ? Vector.magnitude(queryVector) : 0.0;

        // O(1)
        ExecutorService executor;

        // O(1) executor creation (runtime cost depends on JVM)
        if (cfg.useVirtualThreads()) {
            executor = Executors.newVirtualThreadPerTaskExecutor();
        } else {
            executor = Executors.newFixedThreadPool(procs);
        }

        try {
            // O(parts)
            List<Future<PriorityQueue<SearchEngine.Result>>> futures = new ArrayList<>();

            // O(parts)
            for (int p = 0; p < parts; p++) {

                // O(1)
                final int start = p * chunkSize;

                // O(1)
                final int end = Math.min(total, start + chunkSize);

                // O(1)
                if (start >= end) break;

                // O(1) create task submission
                futures.add(executor.submit(() -> {

                    // O(topN) space for local heap
                    // Rationale: min-heap keeps best n for this chunk
                    PriorityQueue<SearchEngine.Result> heap =
                            new PriorityQueue<>(Comparator.comparingDouble(r -> r.score));

                    // Scoring loop:
                    // - total vectors scored across tasks is V
                    // - each score costs O(D)
                    // - heap update costs O(log n)
                    for (int i = start; i < end; i++) {
                        // O(1)
                        Map.Entry<String, double[]> e = entries.get(i);

                        // O(1)
                        String token = e.getKey();

                        // O(1)
                        double[] vec = e.getValue();

                        // O(D) scoring
                        double score = computeScore(vec, token, queryVector, queryMag, precomputedMags, cfg.metric());

                        // O(log n) heap update
                        consider(heap, new SearchEngine.Result(token, score), cfg.topN());
                    }

                    // O(1)
                    return heap;
                }));
            }

            // O(1)
            PriorityQueue<SearchEngine.Result> finalHeap =
                    new PriorityQueue<>(Comparator.comparingDouble(r -> r.score));

            // Merge:
            // - each returned heap contains up to n items
            // - each merge insert is O(log n)
            for (Future<PriorityQueue<SearchEngine.Result>> f : futures) {
                try {
                    // O(1) wait + retrieval cost depends on completion time
                    PriorityQueue<SearchEngine.Result> h = f.get();

                    // O(n * log n) for each heap merge (as bounded by topN)
                    while (!h.isEmpty()) {
                        consider(finalHeap, h.poll(), cfg.topN());
                    }
                } catch (ExecutionException e) {
                    // O(1)
                    Throwable cause = (e.getCause() != null) ? e.getCause() : e;
                    if (cause instanceof RuntimeException re) throw re;
                    throw new RuntimeException(cause);
                }
            }

            // Convert final heap to a descending list:
            // - poll topN times costs O(n log n)
            // - reverse list costs O(n)
            List<SearchEngine.Result> out = new ArrayList<>();
            while (!finalHeap.isEmpty()) {
                // O(log n) per poll
                out.add(finalHeap.poll());
            }

            // O(n)
            Collections.reverse(out);

            // O(1)
            return out;

        } finally {
            // O(1)
            executor.shutdown();
        }
    }

    /**
     * Inserts into a min-heap that stores the top-n results.
     *
     * Big-O:
     * - O(log n) for heap add/poll operations.
     *
     * @param heap min-heap of results
     * @param r candidate result
     * @param topN maximum heap size
     */
    private static void consider(PriorityQueue<SearchEngine.Result> heap, SearchEngine.Result r, int topN) {
        // O(log n) per add/poll
        if (heap.size() < topN) heap.add(r);
        else if (heap.peek().score < r.score) {
            heap.poll();
            heap.add(r);
        }
    }

    /**
     * Computes a similarity/distance score between queryVector and vec.
     *
     * Big-O:
     * - O(D) per scoring call due to dot/product-distance operations.
     *
     * @param vec embedding vector to score
     * @param token token associated with vec (needed for magnitude lookup)
     * @param queryVector query vector
     * @param queryMag precomputed magnitude for query vector (only used for cosine)
     * @param precomputedMags precomputed magnitudes for vec (may be null)
     * @param metric metric choice
     * @return score where higher is better
     */
    private static double computeScore(
            double[] vec,
            String token,
            double[] queryVector,
            double queryMag,
            Map<String, Double> precomputedMags,
            SearchEngine.Metric metric
    ) {
        // O(D) scoring dominates
        return switch (metric) {

            case DOT -> Vector.dot(queryVector, vec); // O(D)

            case COSINE -> {
                // If vecMag is precomputed: O(1) lookup, otherwise magnitude costs O(D)
                double vecMag = (precomputedMags == null)
                        ? Vector.magnitude(vec) // O(D)
                        : precomputedMags.getOrDefault(token, Vector.magnitude(vec)); // O(1) or fallback O(D)

                yield Vector.cosine(queryVector, vec, queryMag, vecMag); // O(D)
            }

            case EUCLIDEAN ->
                    // Negated squared Euclidean so higher is better
                    -Vector.euclideanSquared(queryVector, vec); // O(D)
        };
    }
}