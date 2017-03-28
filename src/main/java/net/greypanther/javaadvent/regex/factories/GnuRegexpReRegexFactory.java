package net.greypanther.javaadvent.regex.factories;

import gnu.regexp.REException;
import gnu.regexp.REMatch;
import net.greypanther.javaadvent.regex.Regex;

import java.util.ArrayList;

import static gnu.regexp.RE.REG_DOT_NEWLINE;

public final class GnuRegexpReRegexFactory extends RegexFactory {

    @Override
    public Regex create(String pattern) {
        final gnu.regexp.RE regexpr;
        try {
            regexpr = new gnu.regexp.RE(pattern, REG_DOT_NEWLINE);
        } catch (REException e) {
            throw new IllegalArgumentException(e);
        }

        return new Regex() {
            @Override
            public boolean containsMatch(String string) {
                return regexpr.isMatch(string);
            }

            @Override
            public Iterable<String[]> getMatches(String string, int[] groups) {
                throw new UnsupportedOperationException();
            }
        };
    }

}
