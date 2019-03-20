# -*- coding: utf-8 -*-

'''
In DNA strings, symbols "A" and "T" are complements of each other, as "C" and "G". You have function with one side of the DNA (string, except for Haskell); you need to get the other complementary side. DNA strand is never empty or there is no DNA at all (again, except for Haskell).

DNA_strand ("ATTGC") # return "TAACG"

DNA_strand ("GTAT") # return "CATA"
'''
def DNA_strand(dna): 
    _str=''   
    for s in dna:
        if(s=='A'):
            _str+='T'
        elif(s=='T'):
            _str+='A'
        elif(s=='C'):
            _str+='G'
        elif(s=='G'):
            _str+='C'
        else:
            pass
    return _str
print(DNA_strand("ATTGC"))