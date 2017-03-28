package net.greypanther.javaadvent.regex.factories;

import net.greypanther.javaadvent.regex.Regex;

import java.util.ArrayList;

public final class ComStevesoftPatRegexFactory extends RegexFactory {

    @Override
    public Regex create(String pattern) {
        final com.stevesoft.pat.Regex regexpr = new com.stevesoft.pat.Regex(pattern);

        return new Regex() {
            @Override
            public boolean containsMatch(String string) {
                try {
                    return regexpr.search(string);
                } catch (StackOverflowError err) {
                    return false;
                }
            }

            @Override
            public Iterable<String[]> getMatches(String string, int[] groups) {
                throw new UnsupportedOperationException();
            }
        };
    }

}
