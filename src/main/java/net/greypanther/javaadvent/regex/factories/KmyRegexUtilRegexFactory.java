package net.greypanther.javaadvent.regex.factories;

import net.greypanther.javaadvent.regex.Regex;

import java.util.ArrayList;

public final class KmyRegexUtilRegexFactory extends RegexFactory {

    @Override
    public Regex create(String pattern) {
        final kmy.regex.util.Regex regexpr = kmy.regex.util.Regex.createRegex(pattern.replace(".", "(.|\n)"));
        return new Regex() {
            @Override
            public boolean containsMatch(String string) {
                return regexpr.matches(string);
            }

            @Override
            public Iterable<String[]> getMatches(String string, int[] groups) {
                throw new UnsupportedOperationException();
            }
        };
    }

}
