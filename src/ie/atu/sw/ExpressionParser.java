package ie.atu.sw;

import java.util.*;

/**
 * ExpressionParser
 *
 * Parses and evaluates vector expressions with operator precedence and scalar support.
 *
 * Supported operators: {@code + - * /} and parentheses.
 * Also supports unary minus.
 *
 * Semantics (element-wise):
 * - If either operand is a scalar, the scalar is broadcast to a vector of length {@code dim}.
 * - If the final expression is scalar-only, it is broadcast to a vector of length {@code dim}.
 *
 * Tokenization and parsing use a shunting-yard algorithm to produce RPN.
 */
public final class ExpressionParser {

    // O(1)
    // Utility class: prevent instantiation.
    private ExpressionParser() { }

    /**
     * Supported operator kinds.
     *
     * NEG is unary negation, represented internally as '~'.
     */
    private enum Op {
        ADD('+', 1, false),
        SUB('-', 1, false),
        MUL('*', 2, false),
        DIV('/', 2, false),
        NEG('~', 3, true); // unary negation, represented internally as '~'

        final char symbol;
        final int precedence;
        final boolean rightAssociative;

        // O(1)
        Op(char symbol, int precedence, boolean rightAssociative) {
            this.symbol = symbol;
            this.precedence = precedence;
            this.rightAssociative = rightAssociative;
        }
    }

    private enum TokenType { NUMBER, WORD, OP, LPAREN, RPAREN }

    /**
     * Token structure used by tokenizer + parser.
     *
     * Big-O: O(1) per token.
     */
    private static final class Token {
        final TokenType type;
        final String text;  // for WORD/OP
        final Op op;        // for OP tokens
        final double number; // for NUMBER tokens

        Token(TokenType type, String text, Op op, double number) {
            this.type = type;
            this.text = text;
            this.op = op;
            this.number = number;
        }

        // O(1)
        static Token number(double v) { return new Token(TokenType.NUMBER, null, null, v); }

        // O(1)
        static Token word(String w) { return new Token(TokenType.WORD, w, null, 0.0); }

        // O(1)
        static Token op(Op op) { return new Token(TokenType.OP, null, op, 0.0); }

        // O(1)
        static Token lparen() { return new Token(TokenType.LPAREN, null, null, 0.0); }

        // O(1)
        static Token rparen() { return new Token(TokenType.RPAREN, null, null, 0.0); }
    }

    /**
     * Value wrapper used during RPN evaluation.
     * Holds either a scalar or a vector.
     */
    private static final class Value {
        final boolean isVector;
        final double[] vec; // valid if isVector
        final double scalar; // valid if !isVector

        private Value(boolean isVector, double[] vec, double scalar) {
            this.isVector = isVector;
            this.vec = vec;
            this.scalar = scalar;
        }

        // O(1)
        static Value vector(double[] v) { return new Value(true, v, 0.0); }

        // O(1)
        static Value scalar(double s) { return new Value(false, null, s); }
    }

    /**
     * Extracts all WORD tokens (non-numeric identifiers) from an expression.
     * Used by streaming mode to know which vectors to load.
     *
     * Big-O:
     * - tokenize: O(L) where L is expression length
     * - scanning tokens: O(T)
     * - storing words: O(W) where W is number of distinct word tokens
     *
     * @param expression expression string to scan
     * @return distinct set of tokens that are treated as WORDs
     * @throws IllegalArgumentException if expression contains unexpected characters
     */
    public static Set<String> extractWords(String expression) {
        // O(1)
        Objects.requireNonNull(expression);

        // O(L) time + O(T) token storage
        List<Token> tokens = tokenize(expression);

        // O(W) space for distinct word tokens
        Set<String> words = new LinkedHashSet<>();
        for (Token t : tokens) {
            // O(1) average add
            if (t.type == TokenType.WORD) words.add(t.text);
        }

        // O(W)
        return words;
    }

    /**
     * Evaluates the expression into a vector of length {@code dim}.
     *
     * Big-O:
     * - tokenization: O(L)
     * - shunting-yard (tokens -> RPN): O(T)
     * - RPN evaluation: O(R * D) where R is the number of binary ops requiring vector work
     *   and D is the embedding dimension.
     *
     * @param expression expression string to evaluate
     * @param wordVectors embedding vectors map (token -> vector)
     * @param dim expected embedding dimension
     * @return resulting vector of length {@code dim}
     * @throws IllegalArgumentException if an unknown token is used or expression is invalid
     */
    public static double[] evaluate(String expression, Map<String, double[]> wordVectors, int dim) {
        // O(1)
        Objects.requireNonNull(expression);
        Objects.requireNonNull(wordVectors);

        // O(1)
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0");

        // O(L) + O(T)
        List<Token> rpn = toRpn(tokenize(expression));

        // O(T) stack usage worst case
        Deque<Value> stack = new ArrayDeque<>();

        // O(T) pass; dominant vector work is O(R * D)
        for (Token t : rpn) {
            switch (t.type) {
                case NUMBER -> {
                    // O(1)
                    stack.push(Value.scalar(t.number));
                }

                case WORD -> {
                    // O(1) average map lookup
                    double[] vec = wordVectors.get(t.text);

                    // O(1)
                    if (vec == null) {
                        throw new IllegalArgumentException("Unknown token in embeddings: '" + t.text + "'");
                    }

                    // O(1)
                    if (vec.length != dim) {
                        throw new IllegalArgumentException(
                                "Embedding dimension mismatch for token '" + t.text
                                        + "' expected " + dim + " got " + vec.length
                        );
                    }

                    // O(1)
                    stack.push(Value.vector(vec));
                }

                case OP -> {
                    if (t.op == Op.NEG) {
                        // O(1)
                        if (stack.isEmpty()) throw new IllegalArgumentException("Missing operand for unary '-'");

                        // O(1)
                        Value a = stack.pop();

                        // O(D) only if operand is a vector; else O(1)
                        stack.push(negate(a));
                    } else {
                        // O(1)
                        if (stack.size() < 2) {
                            throw new IllegalArgumentException("Missing operands for operator '" + t.op.symbol + "'");
                        }

                        // O(1) pops
                        Value b = stack.pop();
                        Value a = stack.pop();

                        // O(D) for element-wise operations
                        stack.push(applyBinary(a, b, t.op, dim));
                    }
                }

                default -> throw new IllegalStateException("Unexpected token in RPN: " + t.type);
            }
        }

        // O(1)
        if (stack.size() != 1) {
            throw new IllegalArgumentException("Invalid expression (stack size " + stack.size() + ")");
        }

        // O(1)
        Value result = stack.pop();

        // O(1) for vector case (return reference)
        if (result.isVector) return result.vec;

        // O(D) for broadcasting scalar-only result
        return Vector.scalarToVector(dim, result.scalar);
    }

    // ---------- Parsing helpers ----------

    /**
     * Converts expression string to tokens.
     *
     * Big-O:
     * - Time: O(L) single forward scan
     * - Space: O(T) for produced tokens
     */
    private static List<Token> tokenize(String expression) {
        // O(L)
        List<Token> tokens = new ArrayList<>();

        // O(1)
        int i = 0;

        // O(1)
        TokenType prevType = null;

        while (i < expression.length()) {
            // O(1)
            char c = expression.charAt(i);

            // O(1) whitespace skip
            if (Character.isWhitespace(c)) {
                i++;
                continue;
            }

            // O(1) parentheses
            if (c == '(') {
                tokens.add(Token.lparen());
                prevType = TokenType.LPAREN;
                i++;
                continue;
            }
            if (c == ')') {
                tokens.add(Token.rparen());
                prevType = TokenType.RPAREN;
                i++;
                continue;
            }

            // O(1) operators
            if (c == '+' || c == '-' || c == '*' || c == '/') {
                // O(1)
                boolean unaryMinus = false;

                if (c == '-') {
                    // O(1)
                    unaryMinus = isUnaryPosition(prevType);

                    if (unaryMinus) {
                        // O(1) lookahead
                        int j = i + 1;

                        if (j < expression.length()
                                && (Character.isDigit(expression.charAt(j)) || expression.charAt(j) == '.')) {
                            // O(k) parse number
                            ParseNumberResult numRes = parseNumber(expression, i + 1);
                            tokens.add(Token.number(-numRes.value)); // O(1)
                            i = numRes.nextIndex; // O(1)
                            prevType = TokenType.NUMBER;
                            continue;
                        } else {
                            // O(1) emit unary NEG
                            tokens.add(Token.op(Op.NEG));
                            prevType = TokenType.OP;
                            i++;
                            continue;
                        }
                    }
                }

                // O(1) map char -> Op
                Op op = switch (c) {
                    case '+' -> Op.ADD;
                    case '-' -> Op.SUB;
                    case '*' -> Op.MUL;
                    case '/' -> Op.DIV;
                    default -> throw new IllegalStateException("Unexpected operator: " + c);
                };

                // O(1)
                tokens.add(Token.op(op));
                prevType = TokenType.OP;
                i++;
                continue;
            }

            // Number literal start
            if (Character.isDigit(c) || c == '.') {
                // O(k)
                ParseNumberResult numRes = parseNumber(expression, i);
                tokens.add(Token.number(numRes.value)); // O(1)
                i = numRes.nextIndex; // O(1)
                prevType = TokenType.NUMBER;
                continue;
            }

            // WORD token: scan until whitespace or delimiters
            // O(|word|), overall contributes to O(L)
            int start = i;
            while (i < expression.length()) {
                char ch = expression.charAt(i);
                if (Character.isWhitespace(ch)
                        || ch == '(' || ch == ')' || ch == '+'
                        || ch == '-' || ch == '*' || ch == '/') {
                    break;
                }
                i++;
            }

            // O(1)
            if (start == i) {
                throw new IllegalArgumentException(
                        "Unexpected character at position " + i + ": '" + c + "'"
                );
            }

            // O(|word|) substring creation
            String word = expression.substring(start, i);

            // O(1)
            tokens.add(Token.word(word));
            prevType = TokenType.WORD;
        }

        // O(T)
        return tokens;
    }

    // O(1)
    private static boolean isUnaryPosition(TokenType prevType) {
        // Unary if start of expression or after an operator or '('
        return prevType == null || prevType == TokenType.OP || prevType == TokenType.LPAREN;
    }

    /**
     * Helper result for parseNumber.
     */
    private static final class ParseNumberResult {
        final double value;
        final int nextIndex;

        ParseNumberResult(double value, int nextIndex) {
            this.value = value;
            this.nextIndex = nextIndex;
        }
    }

    /**
     * Parses a decimal number starting at {@code startIndex}.
     *
     * Big-O: O(k) where k is number literal length.
     */
    private static ParseNumberResult parseNumber(String s, int startIndex) {
        // O(1)
        int i = startIndex;

        // O(1)
        boolean seenDot = false;

        // O(k)
        while (i < s.length()) {
            char c = s.charAt(i);

            if (Character.isDigit(c)) {
                i++;
                continue;
            }
            if (c == '.' && !seenDot) {
                seenDot = true;
                i++;
                continue;
            }
            break;
        }

        // O(k) substring + parseDouble dependency
        String numText = s.substring(startIndex, i);

        double value;
        try {
            // O(k) parseDouble
            value = Double.parseDouble(numText);
        } catch (NumberFormatException nfe) {
            throw new IllegalArgumentException("Invalid number literal: '" + numText + "'");
        }

        // O(1)
        return new ParseNumberResult(value, i);
    }

    /**
     * Shunting-yard algorithm: tokens -> RPN list.
     *
     * Big-O:
     * - Time: O(T) amortized
     * - Space: O(T) for output + operator stack
     */
    private static List<Token> toRpn(List<Token> tokens) {
        // O(T)
        List<Token> output = new ArrayList<>();

        // O(T)
        Deque<Token> ops = new ArrayDeque<>();

        // O(T)
        for (Token t : tokens) {
            switch (t.type) {
                case NUMBER, WORD -> output.add(t); // O(1)

                case OP -> {
                    // O(1)
                    Op o1 = t.op;

                    // Amortized O(T) precedence handling
                    while (!ops.isEmpty() && ops.peek().type == TokenType.OP) {
                        Op o2 = ops.peek().op;

                        // O(1) comparisons
                        boolean higherPrec = o2.precedence > o1.precedence;
                        boolean equalPrecLeftAssoc = (o2.precedence == o1.precedence && !o1.rightAssociative);

                        if (higherPrec || equalPrecLeftAssoc) {
                            // O(1)
                            output.add(ops.pop());
                        } else {
                            break;
                        }
                    }

                    // O(1)
                    ops.push(t);
                }

                case LPAREN -> ops.push(t); // O(1)

                case RPAREN -> {
                    boolean foundLParen = false; // O(1)

                    while (!ops.isEmpty()) {
                        Token top = ops.pop(); // O(1)
                        if (top.type == TokenType.LPAREN) {
                            foundLParen = true; // O(1)
                            break;
                        } else {
                            output.add(top); // O(1)
                        }
                    }

                    if (!foundLParen) {
                        throw new IllegalArgumentException("Mismatched parentheses in expression");
                    }
                }
            }
        }

        // O(T)
        while (!ops.isEmpty()) {
            Token top = ops.pop();
            if (top.type == TokenType.LPAREN || top.type == TokenType.RPAREN) {
                throw new IllegalArgumentException("Mismatched parentheses in expression");
            }
            output.add(top);
        }

        // O(T)
        return output;
    }

    // ----- Evaluation helpers -----

    /**
     * Unary negate.
     *
     * Big-O:
     * - O(D) if argument is a vector
     * - O(1) if argument is scalar
     */
    private static Value negate(Value a) {
        if (a.isVector) {
            // O(D)
            double[] out = new double[a.vec.length];
            for (int i = 0; i < out.length; i++) out[i] = -a.vec[i];
            return Value.vector(out);
        }

        // O(1)
        return Value.scalar(-a.scalar);
    }

    /**
     * Applies a binary operator with scalar broadcast semantics.
     *
     * Big-O:
     * - O(1) when both scalars
     * - O(D) when producing an output vector
     */
    private static Value applyBinary(Value a, Value b, Op op, int dim) {
        // O(1)
        if (!a.isVector && !b.isVector) {
            return Value.scalar(applyScalar(a.scalar, b.scalar, op));
        }

        // O(D) worst-case when broadcasting scalar -> vector
        double[] av = a.isVector ? a.vec : Vector.scalarToVector(dim, a.scalar);
        double[] bv = b.isVector ? b.vec : Vector.scalarToVector(dim, b.scalar);

        // O(D) allocate output
        double[] out = new double[dim];

        // O(D) element-wise operation
        switch (op) {
            case ADD -> {
                for (int i = 0; i < dim; i++) out[i] = av[i] + bv[i];
            }
            case SUB -> {
                for (int i = 0; i < dim; i++) out[i] = av[i] - bv[i];
            }
            case MUL -> {
                for (int i = 0; i < dim; i++) out[i] = av[i] * bv[i];
            }
            case DIV -> {
                for (int i = 0; i < dim; i++) {
                    double denom = bv[i];
                    out[i] = denom == 0.0 ? 0.0 : av[i] / denom;
                }
            }
            default -> throw new IllegalStateException("Unexpected op: " + op);
        }

        // O(1)
        return Value.vector(out);
    }

    /**
     * Applies a scalar operator.
     *
     * Big-O: O(1)
     */
    private static double applyScalar(double a, double b, Op op) {
        // O(1)
        return switch (op) {
            case ADD -> a + b;
            case SUB -> a - b;
            case MUL -> a * b;
            case DIV -> (b == 0.0 ? 0.0 : a / b);
            default -> throw new IllegalStateException("Unexpected scalar op: " + op);
        };
    }
}