package net.greypanther.javaadvent.regex;

public interface Regex {
    boolean containsMatch(String string);
    Iterable<String[]> getMatches(String string, int[] groups);
}
