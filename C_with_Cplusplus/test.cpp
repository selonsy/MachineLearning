#include <iostream>
#include <string>
using namespace std;

void rev_str(char *s,int len)
{
    char ch;
    if ( len > 1)
        {
             ch = *s;
             *s = *(s+len-1);
             *(s+len-1) = ch;
             rev_str(s+1,len-2);
        }
}
int main()
{
    char s[]= "uvxyz";
    rev_str(s,5); 
    cout << s << endl;
}

