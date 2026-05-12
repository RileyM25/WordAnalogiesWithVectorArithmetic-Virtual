package ie.atu.sw;

/**
 * Config
 *
 * Holds application options:
 * - topN: number of best matches to return
 * - metric: similarity/distance metric to use
 * - partitionFactor: number of partitions relative to available processors
 * - useVirtualThreads: whether to use virtual threads for parallel scoring
 *
 * Big-O:
 * - Record fields are stored directly in the immutable record; no algorithmic work here.
 */
public record Config(
        int topN,
        SearchEngine.Metric metric,
        int partitionFactor,
        boolean useVirtualThreads
) {
    /**
     * Convenience factory using sensible defaults.
     *
     * Big-O:
     * - O(1) time and O(1) space because it only creates a new immutable record.
     *
     * @param topN number of results to return
     * @param metric metric to use for scoring
     * @return a new {@code Config} instance with default partitionFactor=2 and useVirtualThreads=true
     */
    public static Config defaults(int topN, SearchEngine.Metric metric) {
        // O(1) create a record with fixed constant defaults
        return new Config(topN, metric, 2, true);
    }
}