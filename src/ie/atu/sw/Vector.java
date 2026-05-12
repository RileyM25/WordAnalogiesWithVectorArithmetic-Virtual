package ie.atu.sw;

import java.util.Arrays;

/**
 * Vector
 *
 * Arithmetic and similarity utilities for vectors.
 *
 * Big-O notes (D = vector length):
 * - Element-wise vector ops (add/sub/mul/div): O(D) time, O(D) space (allocates a new vector)
 * - dot/magnitude/cosine: O(D)
 *
 * Assumption: vectors are same length and non-null.
 */
public final class Vector {

    // O(1)
    private Vector() { /* static utility */ }

    /**
     * Adds two vectors element-wise.
     *
     * Big-O:
     * - Time: O(D)
     * - Space: O(D) for the result vector
     *
     * @param a first vector
     * @param b second vector
     * @return element-wise sum vector
     */
    public static double[] add(double[] a, double[] b) {
        // O(D) time + O(D) space
        int D = a.length;
        double[] r = new double[D];

        // O(D)
        for (int i = 0; i < D; i++) {
            r[i] = a[i] + b[i];
        }

        // O(1)
        return r;
    }

    /**
     * Subtracts vector b from a element-wise (a - b).
     *
     * Big-O:
     * - Time: O(D)
     * - Space: O(D) for the result vector
     *
     * @param a left vector
     * @param b right vector
     * @return element-wise difference vector
     */
    public static double[] sub(double[] a, double[] b) {
        // O(D) time + O(D) space
        int D = a.length;
        double[] r = new double[D];

        // O(D)
        for (int i = 0; i < D; i++) {
            r[i] = a[i] - b[i];
        }

        // O(1)
        return r;
    }

    /**
     * Multiplies two vectors element-wise.
     *
     * Big-O:
     * - Time: O(D)
     * - Space: O(D)
     *
     * @param a left vector
     * @param b right vector
     * @return element-wise product vector
     */
    public static double[] mul(double[] a, double[] b) {
        // O(D) time + O(D) space
        int D = a.length;
        double[] r = new double[D];

        // O(D)
        for (int i = 0; i < D; i++) {
            r[i] = a[i] * b[i];
        }

        // O(1)
        return r;
    }

    /**
     * Divides vector a by b element-wise. If b[i] is 0, outputs 0 at that element.
     *
     * Big-O:
     * - Time: O(D)
     * - Space: O(D)
     *
     * @param a numerator vector
     * @param b denominator vector
     * @return element-wise quotient vector
     */
    public static double[] div(double[] a, double[] b) {
        // O(D) time + O(D) space
        int D = a.length;
        double[] r = new double[D];

        // O(D)
        for (int i = 0; i < D; i++) {
            double denom = b[i];
            r[i] = denom == 0.0 ? 0.0 : a[i] / denom;
        }

        // O(1)
        return r;
    }

    /**
     * Scales vector a by scalar s (broadcast multiplication).
     *
     * Big-O:
     * - Time: O(D)
     * - Space: O(D)
     *
     * @param a vector to scale
     * @param s scalar multiplier
     * @return scaled vector
     */
    public static double[] scale(double[] a, double s) {
        // O(D) time + O(D) space
        int D = a.length;
        double[] r = new double[D];

        // O(D)
        for (int i = 0; i < D; i++) {
            r[i] = a[i] * s;
        }

        // O(1)
        return r;
    }

    /**
     * Broadcasts a scalar into a vector of length {@code dim}, filled with {@code s}.
     *
     * Big-O:
     * - Time: O(D)
     * - Space: O(D)
     *
     * @param dim length of output vector
     * @param s scalar value to fill
     * @return vector filled with s
     */
    public static double[] scalarToVector(int dim, double s) {
        // O(D) time to fill + O(D) space for the result
        double[] r = new double[dim];

        // O(D)
        Arrays.fill(r, s);

        // O(1)
        return r;
    }

    /**
     * Computes the dot product of two vectors.
     *
     * Big-O:
     * - Time: O(D)
     * - Extra space: O(1)
     *
     * @param a first vector
     * @param b second vector
     * @return dot product
     */
    public static double dot(double[] a, double[] b) {
        // O(D) time, O(1) extra space
        double sum = 0.0;

        // O(D)
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }

        // O(1)
        return sum;
    }

    /**
     * Computes squared Euclidean distance between two vectors.
     *
     * Big-O:
     * - Time: O(D)
     * - Extra space: O(1)
     *
     * @param a first vector
     * @param b second vector
     * @return squared Euclidean distance
     */
    public static double euclideanSquared(double[] a, double[] b) {
        // O(D) time, O(1) extra space
        double sum = 0.0;

        // O(D)
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }

        // O(1)
        return sum;
    }

    /**
     * Computes the magnitude (L2 norm) of a vector.
     *
     * Big-O:
     * - Time: O(D)
     * - Extra space: O(1)
     *
     * @param a vector
     * @return vector magnitude
     */
    public static double magnitude(double[] a) {
        // O(D) time, O(1) extra space
        double sum = 0.0;

        // O(D)
        for (double v : a) {
            sum += v * v;
        }

        // O(1)
        return Math.sqrt(sum);
    }

    /**
     * Computes cosine similarity between two vectors.
     * If either vector has magnitude 0, returns 0.0.
     *
     * Big-O:
     * - Time: O(D)
     * - Extra space: O(1)
     *
     * @param a first vector
     * @param b second vector
     * @param magA precomputed magnitude of a
     * @param magB precomputed magnitude of b
     * @return cosine similarity in [-1, 1]
     */
    public static double cosine(double[] a, double[] b, double magA, double magB) {
        // O(1) checks
        if (magA == 0.0 || magB == 0.0) return 0.0;

        // O(D) for dot; O(1) for division
        return dot(a, b) / (magA * magB);
    }
}