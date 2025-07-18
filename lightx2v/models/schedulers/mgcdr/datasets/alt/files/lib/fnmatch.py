"""Filename matching with shell patterns.

fnmatch(FILENAME, PATTERN) matches according to the local convention.
fnmatchcase(FILENAME, PATTERN) always takes case in account.

The functions operate by translating the pattern into a regular
expression.  They cache the compiled regular expressions for speed.

The function translate(PATTERN) returns a regular expression
corresponding to PATTERN.  (It does not compile it.)
"""
"""Compared with the standard library, syntax '{seq1,seq2}' is supported"""

import functools
import os
import re
from typing import Callable, List, Match, Optional


def fnmatch(name: str, pat: str) -> bool:
    """Test whether FILENAME matches PATTERN.

    Patterns are Unix shell style:

    *               matches any characters but '/'
    **              matches everything
    ?               matches any single character
    [seq]           matches any character in seq
    [!seq]          matches any char not in seq
    {seq1,seq2}     matches seq1 or seq2

    An initial period in FILENAME is not special.
    Both FILENAME and PATTERN are first case-normalized
    if the operating system requires it.
    If you don't want this, use fnmatchcase(FILENAME, PATTERN).
    """
    name = os.path.normcase(name)
    pat = os.path.normcase(pat)
    return fnmatchcase(name, pat)


def fnmatchcase(name: str, pat: str) -> bool:
    """Test whether FILENAME matches PATTERN, including case.

    This is a version of fnmatch() which doesn't case-normalize
    its arguments.
    """
    match = _compile_pattern(pat)
    return match(name) is not None


@functools.lru_cache(maxsize=256, typed=True)
def _compile_pattern(pat: str) -> Callable[[str], Optional[Match[str]]]:
    res = translate(pat)
    return re.compile(res).match


def filter(names: List[str], pat: str) -> List[str]:
    """Return the subset of the list NAMES that match PAT."""
    result = []
    pat = os.path.normcase(pat)
    match = _compile_pattern(pat)
    for name in names:
        if match(os.path.normcase(name)):
            result.append(name)
    return result


def _compat(res: str) -> str:
    return r'(?s:%s)\Z' % res


def translate(pat: str) -> str:
    """Translate a shell PATTERN to a regular expression.

    There is no way to quote meta-characters.
    """

    i, n = 0, len(pat)
    res = ''
    while i < n:
        c = pat[i]
        i = i + 1
        if c == '*':
            j = i
            while j < n and pat[j] == '*':
                j = j + 1
            if j > i:
                if (j < n and pat[j] == '/') and (i <= 1 or pat[i - 2] == '/'):
                    # hit /**/ instead of /seq**/
                    j = j + 1
                    res = res + r'(.*/)?'
                else:
                    res = res + r'.*'
            else:
                res = res + r'[^/]*'
            i = j
        elif c == '?':
            res = res + r'.'
        elif c == '[':
            j = i
            if j < n and pat[j] == '!':
                j = j + 1
            if j < n and pat[j] == ']':
                j = j + 1
            while j < n and pat[j] != ']':
                j = j + 1
            if j >= n:
                res = res + r'\['
            else:
                stuff = pat[i:j].replace('\\', r'\\')
                i = j + 1
                if stuff[0] == '!':
                    stuff = r'^' + stuff[1:]
                elif stuff[0] == '^':
                    stuff = '\\' + stuff
                res = r'%s[%s]' % (res, stuff)
        elif c == '{':
            j = i
            if j < n and pat[j] == '}':
                j = j + 1
            while j < n and pat[j] != '}':
                j = j + 1
            if j >= n:
                res = res + r'\{'
            else:
                stuff = pat[i:j].replace('\\', r'\\')
                stuff = r'|'.join(map(re.escape, stuff.split(',')))
                res = r'%s(%s)' % (res, stuff)
                i = j + 1
        else:
            res = res + re.escape(c)
    return _compat(res)
