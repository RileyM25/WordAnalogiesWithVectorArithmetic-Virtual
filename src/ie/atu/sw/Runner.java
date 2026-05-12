package ie.atu.sw;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

/**
 * Runner
 *
 * Menu-driven CLI that gathers user inputs and calls:
 * - EmbeddingLoader
 * - ExpressionParser
 * - SearchEngine
 *
 * This class focuses on UI only (high cohesion).
 */
public class Runner {

    /**
     * Cleans user input that may include surrounding quotes (common on Windows CMD).
     *
     * Examples:
     * - {@code "C:\path\file.txt"} -> {@code C:\path\file.txt}
     * - {@code C:\path\file.txt} -> unchanged
     *
     * Big-O:
     * - Time: O(L) due to trim + possible substring
     * - Space: O(L) for the trimmed/substring string
     *
     * @param s user input string
     * @return cleaned string; returns null if input is null
     */
    private static String cleanPathInput(String s) {
        // O(1) null check
        if (s == null) return null;

        // O(L) trim
        s = s.trim();

        // O(1) quote check + O(L) substring trim
        if (s.length() >= 2 && s.charAt(0) == '"' && s.charAt(s.length() - 1) == '"') {
            return s.substring(1, s.length() - 1).trim();
        }

        // O(1)
        return s;
    }

    /**
     * Program entry point for the CLI.
     *
     * Big-O:
     * - O(1) per menu rendering/choice handling (ignoring user typing time)
     * - Dominated by {@link #runOnce(String, String, String, String, int, String)} when the user selects Run.
     *
     * @param args command-line arguments (unused)
     * @throws Exception if underlying validation or file/search operations fail
     */
    public static void main(String[] args) throws Exception {
        // O(1) init; space excluding user input
        Scanner sc = new Scanner(System.in, StandardCharsets.UTF_8);

        String embedPath = null;
        String expression = null;
        String metric = "cosine";     // cosine | dot | euclidean
        String mode = "inmemory";    // inmemory | stream
        int topN = 10;
        String outPath = "./out.txt";

        while (true) {
            printHeader();

            // O(1) printing and input prompt
            System.out.print(ConsoleColour.BLACK_BOLD_BRIGHT);
            System.out.print("Select Option [1-6]>");
            System.out.println();
            System.out.print(ConsoleColour.YELLOW);

            // O(|line|) to read + trim one input line
            String choice = sc.nextLine().trim();

            // O(1) branch selection
            switch (choice) {
                case "1" -> {
                    // O(|path|) read input
                    System.out.print("Enter Path to Embeddings File> ");
                    String cleaned = cleanPathInput(sc.nextLine().trim());

                    // O(1)
                    if (cleaned == null || cleaned.isEmpty()) embedPath = null;
                    else embedPath = cleaned;
                }

                case "2" -> {
                    // O(|expr|) read input
                    System.out.print("Enter Vector Operation/Expression (e.g., irish - whiskey + vodka)> ");
                    String expr = sc.nextLine().trim();

                    // O(1)
                    expression = expr.isEmpty() ? null : expr;
                }

                case "3" -> {
                    // O(1) menu prompts + O(|inputs|) parsing
                    System.out.println("Configure Options:");

                    // O(|input|)
                    System.out.print("Metric [cosine|dot|euclidean] (default cosine)> ");
                    String m = sc.nextLine().trim();
                    if (!m.isEmpty()) metric = m.toLowerCase(Locale.ROOT);

                    // O(|input|)
                    System.out.print("Top N results (default 10)> ");
                    String n = sc.nextLine().trim();
                    if (!n.isEmpty()) topN = Integer.parseInt(n);

                    // O(|input|)
                    System.out.print("Mode [inmemory|stream] (default inmemory)> ");
                    String mm = sc.nextLine().trim();
                    if (!mm.isEmpty()) mode = mm.toLowerCase(Locale.ROOT);
                }

                case "4" -> {
                    // O(|path|) read input
                    System.out.print("Specify Output File (default ./out.txt)> ");
                    String cleaned = cleanPathInput(sc.nextLine().trim());

                    // O(1)
                    if (cleaned != null && !cleaned.isEmpty()) outPath = cleaned;
                }

                case "5" -> {
                    // O(1) validation
                    if (embedPath == null || expression == null) {
                        System.out.println("You must provide both embeddings file path and an expression.");
                        pause(sc);
                        break;
                    }

                    // Dominant runtime is runOnce
                    runOnce(embedPath, expression, metric, mode, topN, outPath);
                    pause(sc);
                }

                case "6" -> {
                    // O(1)
                    System.out.println("Quitting.");
                    return;
                }

                default -> {
                    // O(1)
                    System.out.println("Invalid option. Try again.");
                    pause(sc);
                }
            }
        }
    }

    /**
     * Prints the CLI header UI.
     *
     * Big-O:
     * - O(1) time and O(1) space (constant number of print statements)
     */
    private static void printHeader() {
        // O(1)
        System.out.println(ConsoleColour.RED);
        System.out.println("************************************************************");
        System.out.println(ConsoleColour.RESET); // ensures emoji isn't tinted

        // O(1) fixed strings
        System.out.println("🧠 Word Analogies with Vector Arithmetic & Virtual Threads");
        System.out.println("************************************************************");
        System.out.println("🗂️ (1) Enter Path to Embeddings File");
        System.out.println("✍️ (2) Enter Vector Operation / Expression");
        System.out.println("⚙️ (3) Configure Options");
        System.out.println("💾 (4) Specify Output File (default: ./out.txt)");
        System.out.println("▶️ (5) Run");
        System.out.println("❔ (6) Quit");
        System.out.println();
    }

    /**
     * Runs the full pipeline:
     * - validates inputs
     * - loads embeddings (in-memory or stream)
     * - evaluates the query expression to a vector
     * - searches for top-N using the chosen metric
     * - writes results to file
     *
     * Big-O (dominant term):
     * - In-memory: O(V * D) dominated by embedding loading + scoring/search
     * - Stream:    O(V * D) dominated by scanning + scoring/search
     *
     * Where:
     * - V = number of vectors in embeddings
     * - D = embedding dimension
     *
     * @param embedPathStr embeddings file path as string
     * @param expression expression to evaluate (e.g., king - man + woman)
     * @param metricStr metric name: cosine|dot|euclidean
     * @param modeStr mode name: inmemory|stream
     * @param topN number of results to output
     * @param outPathStr output file path
     * @throws Exception if any stage fails (file errors, missing tokens, invalid expressions, etc.)
     */
    private static void runOnce(
            String embedPathStr,
            String expression,
            String metricStr,
            String modeStr,
            int topN,
            String outPathStr
    ) throws Exception {

        // O(1)
        Path embedPath = Paths.get(embedPathStr);
        Path outPath = Paths.get(outPathStr);

        // O(1) (filesystem metadata check treated constant for Big-O)
        if (!Files.exists(embedPath)) {
            throw new IllegalArgumentException("Embeddings file not found: " + embedPath.toAbsolutePath());
        }

        // O(1)
        if (topN <= 0) {
            throw new IllegalArgumentException("topN must be > 0");
        }

        // O(1) mapping input string to enum
        SearchEngine.Metric metric = switch (metricStr.toLowerCase(Locale.ROOT)) {
            case "cosine" -> SearchEngine.Metric.COSINE;
            case "dot" -> SearchEngine.Metric.DOT;
            case "euclidean" -> SearchEngine.Metric.EUCLIDEAN;
            default -> throw new IllegalArgumentException("Unknown metric: " + metricStr);
        };

        // O(1)
        int size = 100;

        // O(1)
        double[] queryVector;

        // O(1) fixed-size progress loops (ignoring Thread.sleep time)
        printProgress(1, size);
        for (int i = 2; i <= 25; i++) {
            printProgress(i, size);
            Thread.sleep(5);
        }

        if (modeStr.equalsIgnoreCase("inmemory")) {
            // O(V * D) time, O(V * D) space (delegated to loadAll)
            Map<String, double[]> embeddings = EmbeddingLoader.loadAll(embedPath);

            // O(1)
            if (embeddings.isEmpty()) throw new IllegalArgumentException("No embeddings loaded.");

            // O(1) vector dimension from first embedding
            int dim = embeddings.values().iterator().next().length;

            // O(1) fixed progress work
            for (int i = 26; i <= 45; i++) {
                printProgress(i, size);
                Thread.sleep(5);
            }

            // O(L + R*D) approx. delegated to ExpressionParser.evaluate
            queryVector = ExpressionParser.evaluate(expression, embeddings, dim);

            // O(V * D) for cosine precompute, else O(1)
            Map<String, Double> precomputedMags = null;
            if (metric == SearchEngine.Metric.COSINE) {
                // O(V * D) each magnitude costs O(D)
                precomputedMags = new HashMap<>(embeddings.size());
                for (Map.Entry<String, double[]> e : embeddings.entrySet()) {
                    precomputedMags.put(e.getKey(), Vector.magnitude(e.getValue())); // O(D)
                }
            }

            // O(1) fixed progress work
            for (int i = 46; i <= 80; i++) {
                printProgress(i, size);
                Thread.sleep(5);
            }

            // Dominant: O(V * D) scoring + heap/merge overhead (delegated)
            List<SearchEngine.Result> results =
                    SearchEngine.searchInMemory(embeddings, precomputedMags, queryVector, metric, topN);

            // O(1) fixed progress work
            for (int i = 81; i <= 95; i++) {
                printProgress(i, size);
                Thread.sleep(5);
            }

            // O(n) write results (n <= topN)
            writeResults(outPath, results);

        } else if (modeStr.equalsIgnoreCase("stream")) {
            // O(L) delegated to ExpressionParser.extractWords
            Set<String> required = ExpressionParser.extractWords(expression);

            // O(1)
            if (required.isEmpty()) throw new IllegalArgumentException("Expression contains no word tokens.");

            // O(K) map for needed vectors (vectors stored cost O(K*D))
            Map<String, double[]> needed = new HashMap<>();

            // O(V * D) worst-case to detect dimension (delegated to detectDimension)
            int detected = EmbeddingLoader.detectDimension(embedPath);

            // O(1)
            if (detected <= 0) throw new IllegalArgumentException("Could not detect embedding dimension.");

            // O(1) capture final dim for lambda
            final int dimDetected = detected;

            // O(V * D) scan; O(K * D) space for required vectors stored
            EmbeddingLoader.stream(embedPath, (token, vec) -> {
                // O(1) contains average + O(1) length check; storing reference is O(1)
                if (required.contains(token) && vec.length == dimDetected) {
                    needed.put(token, vec);
                }
            });

            // O(1)
            if (dimDetected <= 0) throw new IllegalArgumentException("Could not determine embedding dimension (stream mode).");

            // O(K) average for set ops
            if (!needed.keySet().containsAll(required)) {
                Set<String> missing = required;
                missing.removeAll(needed.keySet());
                throw new IllegalArgumentException("Missing embeddings for tokens: " + missing);
            }

            // O(1) fixed progress work
            for (int i = 56; i <= 70; i++) {
                printProgress(i, size);
                Thread.sleep(5);
            }

            // O(L + R*D) approx. delegated to ExpressionParser.evaluate
            queryVector = ExpressionParser.evaluate(expression, needed, dimDetected);

            // O(1) fixed progress work
            for (int i = 71; i <= 90; i++) {
                printProgress(i, size);
                Thread.sleep(5);
            }

            // Dominant: O(V * D) scan + O(V log n) heap updates (delegated)
            List<SearchEngine.Result> results =
                    SearchEngine.searchStreaming(embedPath, queryVector, metric, topN);

            // O(1) fixed progress work
            for (int i = 91; i <= 95; i++) {
                printProgress(i, size);
                Thread.sleep(5);
            }

            // O(n) write results (n <= topN)
            writeResults(outPath, results);

        } else {
            // O(1)
            throw new IllegalArgumentException("Unknown mode: " + modeStr + " (use inmemory or stream)");
        }

        // O(1)
        System.out.println("\nDone. Results written to: " + outPath.toAbsolutePath());
    }

    /**
     * Writes top-N search results to a UTF-8 text file.
     *
     * Big-O:
     * - Time: O(n) where n = results.size()
     * - Space: O(1) extra space (ignoring output stream buffers)
     *
     * @param outPath output file path
     * @param results list of results
     * @throws IOException if writing fails
     */
    private static void writeResults(Path outPath, List<SearchEngine.Result> results) throws IOException {
        // Delegate to IOManager to avoid duplicated file-writing logic.
        // O(1) metadata + O(n) write loop happens inside IOManager.writeTopN(...) [3]
        IOManager.writeTopN(outPath, results);
    }

    /**
     * Waits for user to press Enter.
     *
     * Big-O:
     * - O(1) algorithmic work (user typing time excluded)
     *
     * @param sc scanner reading from standard input
     */
    private static void pause(Scanner sc) {
        // O(1)
        System.out.println("Press Enter to continue...");
        // O(1) algorithmically; user input time excluded
        sc.nextLine();
    }

    /**
     * Prints a terminal progress bar using emoji characters.
     *
     * Big-O:
     * - O(1) time because bar width is constant (size = 30)
     *
     * @param index current progress index
     * @param total total progress value
     */
    public static void printProgress(int index, int total) {
        // O(1)
        if (index > total) return;

        int size = 30; // O(1) constant width bar
        String done = "✅";
        String todo = "⬜";

        // O(1) computations
        int complete = (100 * index) / total;
        int completeLen = size * complete / 100;

        // O(1) reset to avoid tinting emojis
        System.out.print(ConsoleColour.RESET);

        // O(1) fixed loop
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < size; i++) {
            sb.append((i < completeLen) ? done : todo);
        }

        // O(1) output
        System.out.print("\r" + sb + "] " + complete + "%");
        if (complete == 100) System.out.println("\n");
    }
}