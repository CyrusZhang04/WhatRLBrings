# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import re, json
# Anomaly definition (from task): any of
#  (1) length < 50 or length == max_tokens (the generation limit, 8192)
#  (2) text does not contain "\boxed"
#  (3) in last 3000 chars, the same 1-3 char string repeats consecutively >= 26 times
REP = re.compile(r'(.{1,3}?)\1{25,}', re.S)
MAXTOK = 8192
def flags(text, length, maxtok=MAXTOK):
    f_len = (length < 50) or (length == maxtok)
    f_box = ('\\boxed' not in text)
    f_rep = REP.search(text[-3000:]) is not None
    return f_len, f_box, f_rep
def is_anom(text, length, maxtok=MAXTOK):
    a,b,c = flags(text,length,maxtok)
    return a or b or c
