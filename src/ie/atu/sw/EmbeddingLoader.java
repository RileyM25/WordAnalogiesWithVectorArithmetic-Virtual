package ie.atu.sw;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * EmbeddingLoader
 *
 * Loads word embeddings from a text file and provides:
 * - {@link #loadAll(Path)}: load the full map into memory
 * - {@link #detectDimension(Path)}: detect embedding dimension by scanning
 * - {@link #stream(Path, BiConsumerWithIOException)}: stream only matching embeddings to a consumer
 *
 * Expected file format (per line):
 * token val1 val2 ... valD
 * (commas are tolerated and normalized to spaces)
 */
public final class EmbeddingLoader {

    /**
     * Prevent instantiation.
     *
     * Big-O: O(1)
     */
    private EmbeddingLoader() { }

    /**
     * Loads all embeddings into memory.
     *
     * Big-O (lecture variables):
     * - Let V = number of embedding lines (vectors)
     * - Let D = embedding dimension
     * - Let L = total number of characters processed
     * Time: O(V * D) (dominant cost is parsing D doubles per vector)
     * Space: O(V * D) (stores all vectors)
     *
     * @param path path to embeddings file
     * @return map of token -> embedding vector
     * @throws IOException if reading the file fails
     */
    public static Map<String, double[]> loadAll(Path path) throws IOException {
        // O(1)
        Map<String, double[]> map = new LinkedHashMap<>();

        // O(L) total characters read; per-vector parsing is O(D)
        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            int expectedDim = -1;
            long lineNo = 0;

            while ((line = br.readLine()) != null) { // O(L)
                // O(|line|) trim; asymptotically dominated by parsing doubles
                lineNo++;
                line = line.trim();
                if (line.isEmpty()) continue;

                // O(D) parsing of token + D doubles (or O(|line|) on malformed)
                ParsedLine parsed = parseLine(line, lineNo);
                if (parsed == null) continue;

                // O(1)
                if (expectedDim == -1) expectedDim = parsed.vec.length;

                // O(1)
                if (parsed.vec.length != expectedDim) {
                    // O(1)
                    System.err.printf(
                            "Dimension mismatch at line %d (expected %d, got %d). Skipping.%n",
                            lineNo, expectedDim, parsed.vec.length
                    );
                    continue;
                }

                // O(1) average map insertion by token
                map.put(parsed.token, parsed.vec);
            }
        }

        // O(1)
        return map;
    }

    /**
     * Detects the embedding dimension by scanning the first valid line.
     *
     * Big-O:
     * - Worst case scans up to V lines
     * - Each valid parse costs O(D)
     *
     * Time: O(V * D) worst-case, typically O(D) to the first good line
     * Space: O(1) extra
     *
     * @param path path to embeddings file
     * @return detected dimension, or -1 if no valid line is found
     * @throws IOException if reading the file fails
     */
    public static int detectDimension(Path path) throws IOException {
        // O(1)
        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            long lineNo = 0;

            while ((line = br.readLine()) != null) { // O(L)
                // O(|line|) trim
                lineNo++;
                line = line.trim();
                if (line.isEmpty()) continue;

                // O(D) if parse succeeds, still dominated by parsing
                ParsedLine parsed = parseLine(line, lineNo);
                if (parsed != null) return parsed.vec.length;
            }
        }

        // O(1)
        return -1;
    }

    /**
     * Streams embeddings line-by-line and sends parsed results to the consumer.
     * Filters out lines whose vector dimension doesn't match the first valid dimension.
     *
     * Big-O:
     * - Let V = number of lines
     * Time: O(V * D) (reads and parses doubles for matching-dimension vectors)
     * Space: O(1) extra (not counting consumer storage)
     *
     * @param path path to embeddings file
     * @param consumer consumer callback invoked for each valid token/vector
     * @throws IOException if reading the file or consumer callback fails
     */
    public static void stream(
            Path path,
            BiConsumerWithIOException<String, double[]> consumer
    ) throws IOException {

        // O(1)
        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            int expectedDim = -1;
            long lineNo = 0;

            while ((line = br.readLine()) != null) { // O(L)
                // O(|line|) trim; dominated by parse
                lineNo++;
                line = line.trim();
                if (line.isEmpty()) continue;

                // O(D) parse attempt
                ParsedLine parsed = parseLine(line, lineNo);
                if (parsed == null) continue;

                // O(1) capture expected dimension from first valid line
                if (expectedDim == -1) expectedDim = parsed.vec.length;

                // O(1) mismatch filter
                if (parsed.vec.length != expectedDim) continue;

                // O(1) call to consumer (consumer cost depends on caller)
                consumer.accept(parsed.token, parsed.vec);
            }
        }
    }

    // ---------- Parsing helpers ----------

    /**
     * Parses one line into a token and vector.
     *
     * Big-O:
     * - Let D be vector dimension for this line.
     * Time: O(D) for splitting and parsing D doubles
     * Space: O(D) for the resulting vector array
     *
     * @param line input line
     * @param lineNo line number for error reporting
     * @return parsed line, or null if malformed
     */
    private static ParsedLine parseLine(String line, long lineNo) {
        // O(|line|) to create normalized string
        String normalized = line.replace(',', ' ').trim();

        // O(|line|) split creates substrings/array
        String[] parts = normalized.split("\\s+");

        // O(1) check
        if (parts.length < 2) {
            // O(1) error reporting
            System.err.printf("Skipping malformed line %d: '%s'%n", lineNo, line);
            return null;
        }

        // O(1)
        String token = parts[0];

        // O(1)
        int dim = parts.length - 1;

        // O(D) array allocation
        double[] vec = new double[dim];

        // O(D) parse D doubles
        for (int i = 0; i < dim; i++) {
            try {
                // O(1) per Double.parseDouble (amortized) + dependency on substring length
                vec[i] = Double.parseDouble(parts[i + 1]);
            } catch (NumberFormatException nfe) {
                // O(1) error reporting
                System.err.printf(
                        "Bad double at line %d, col %d: '%s'%n",
                        lineNo, i + 1, parts[i + 1]
                );
                return null;
            }
        }

        // O(1)
        return new ParsedLine(token, vec);
    }

    // O(1)
    // Immutable parsed representation.
    private record ParsedLine(String token, double[] vec) { }

    /**
     * Consumer functional interface used by {@link #stream(Path, BiConsumerWithIOException)}.
     *
     * Big-O:
     * - No algorithmic complexity by itself.
     *
     * @param <K> type of the token key
     * @param <V> type of the vector value
     */
    @FunctionalInterface
    public interface BiConsumerWithIOException<K, V> {
        /**
         * Accepts the key/value pair produced by the stream.
         *
         * @param k token
         * @param v embedding vector
         * @throws IOException if the consumer fails
         */
        void accept(K k, V v) throws IOException;
    }
}