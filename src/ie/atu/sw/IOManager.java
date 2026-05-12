package ie.atu.sw;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Locale;

/**
 * IOManager
 *
 * Handles output file writing of top-N results (UTF-8).
 */
public final class IOManager {

    /**
     * Prevent instantiation.
     *
     * Big-O: O(1)
     */
    private IOManager() { }

    /**
     * Writes the top-N results to a UTF-8 file.
     *
     * Output format (per line):
     * rank<TAB>token<TAB>score
     *
     * Big-O:
     * - Let n = number of results written (typically n = topN)
     * - Time: O(n) because we iterate through the result list once
     * - Space: O(1) extra space (ignoring the provided `results` list and output stream buffers)
     *
     * @param outPath output file path
     * @param results list of results to write (typically <= topN)
     * @throws IOException if writing fails
     */
    public static void writeTopN(Path outPath, List<SearchEngine.Result> results) throws IOException {
        // O(1) metadata check for parent directory
        Path parent = outPath.toAbsolutePath().getParent();
        if (parent != null) {
            // O(1) from algorithmic perspective (filesystem cost depends on environment)
            Files.createDirectories(parent);
        }

        // O(n) time for writing n lines
        try (BufferedWriter bw = Files.newBufferedWriter(
                outPath,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
        )) {
            // O(1) write header
            bw.write("rank\ttoken\tscore\n");

            // O(n) loop over results
            for (int i = 0; i < results.size(); i++) {
                // O(1) per list access
                SearchEngine.Result r = results.get(i);

                // O(|token|) for replace; token length is usually small, dominated by O(n)
                String safeToken = r.token.replace('\t', ' ').replace('\n', ' ');

                // O(1) numeric formatting for fixed "%.6f"
                bw.write((i + 1) + "\t" + safeToken + "\t"
                        + String.format(Locale.ROOT, "%.6f", r.score) + "\n");
            }
        }
    }
}