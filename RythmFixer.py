from typing import List
from sympy import *
import re


_time_sig_regex = re.compile(r'\d+')

def _get_time_signature_frac(timeSignature : str) -> Rational:
    timeSignature = timeSignature[14:]
    if 'C' in timeSignature:
        return Rational(4,4)
    nums = _time_sig_regex.findall(timeSignature)
    return Rational(nums[0], nums[1])

_note_relative_values = (
    ('quadruple',               4),
    ('double',                  2),
    ('whole',                   1),
    ('half',                    Rational(1,2)),
    ('quarter',                 Rational(1,4)),
    ('eight',                   Rational(1,8)),
    ('sixteenth',               Rational(1,16)),
    ('thirty_second',           Rational(1,32)),
    ('sixty_fourth',            Rational(1,64)),
    ('hundred_twenty_eighth',   Rational(1,128))
)

def _get_relative_val(token : str) -> Rational:
    for note_name, duration in _note_relative_values:
         if note_name in token:
             if token[len(token) - 1] == '.':
                 duration += (Rational(1,2) * duration)
             return duration
    return 0

def fix_rythm(tokens : List[str]) -> List[str]:
    fixed_tokens_no_barlines = [ x for x in tokens if x != 'barline' ]
    fixed_tokens = []
    i = 0
    time_signature = None
    while i < len(fixed_tokens_no_barlines):
        fixed_tokens.append(fixed_tokens_no_barlines[i])
        if 'timeSignature' in fixed_tokens_no_barlines[i]:
            time_signature = _get_time_signature_frac(fixed_tokens_no_barlines[i])
            i += 1
            break
        i += 1
    if time_signature != None:
        bar_relative_val = 0
        while i < len(fixed_tokens_no_barlines):
            relative_val = _get_relative_val(fixed_tokens_no_barlines[i])
            fixed_tokens.append(fixed_tokens_no_barlines[i])
            bar_relative_val += relative_val
            if (bar_relative_val >= time_signature):
                fixed_tokens.append('barline')
                bar_relative_val = 0
            i += 1
    return fixed_tokens
