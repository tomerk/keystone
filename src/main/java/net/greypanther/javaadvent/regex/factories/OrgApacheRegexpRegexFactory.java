package net.greypanther.javaadvent.regex.factories;

import net.greypanther.javaadvent.regex.Regex;

import java.util.ArrayList;

public class OrgApacheRegexpRegexFactory extends RegexFactory {
    @Override
    public Regex create(String pattern) {
        org.apache.regexp.RE re;
        try {
            re = new org.apache.regexp.RE(pattern);
        } catch (Exception | Error ex) {
            // don't match anything
            re = new org.apache.regexp.RE(".^");
        }

        final org.apache.regexp.RE regexpr = re;
        return new Regex() {
            @Override
            public boolean containsMatch(String string) {
                try {
                    return regexpr.match(string);
                } catch (Exception | Error ex) {
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
