#include <stdio.h>

/*
汉诺塔，将盘子从start杆移动到end杆
第一步，将start杆上n-1个饼借助end杆移动到swap杆
第二步，将start杆上最后一个饼移动到end杆上，输出
第三步，将swap杆上n-1个饼移借助start动到end上
*/
void han(char start, char end, char swap,int n){
	/*
	 * start: 起始杆
	 * end: 结束杆
	 * swap: 辅助杆
	 */
    if(n<=0)
        return;
    han(start, swap, end,n-1);
    printf("move from %c to %c\n",start, end);
    han(swap, end, start,n-1);
}
int main(){
    int n;
    while(scanf("%d",&n)!=EOF){
        if(n<=0){
            printf("Input Error");
            continue;
        }
        han('A','C','B',n);
    }
    return 0;
}
