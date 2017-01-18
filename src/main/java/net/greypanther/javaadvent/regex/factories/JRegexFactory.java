package net.greypanther.javaadvent.regex.factories;

import jregex.REFlags;
import net.greypanther.javaadvent.regex.Regex;

public final class JRegexFactory extends RegexFactory {

    @Override
    public Regex create(String pattern) {
        final jregex.Pattern regexpr = new jregex.Pattern(pattern, REFlags.DOTALL);

        return new Regex() {
            @Override
            public boolean containsMatch(String string) {
                return regexpr.matcher(string).matches();
            }
        };
    }

}
