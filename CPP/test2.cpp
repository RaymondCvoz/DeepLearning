#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <climits>

using namespace std;


class Solution {
public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        const int lens = 2e4 + 3;
        int field[lens];
        memset(field,0,sizeof(field));
        for(int i = 0;i < flowerbed.size();i++)
        {
            field[i + 1] = flowerbed[i];
        }
        for(int i = 1;i <= flowerbed.size();i++)
        {
            if(field[i] == 0 && field[i - 1] == 0 && field[i + 1] == 0)
            {
                field[i] = 1;
                n --;
            }
            if(n <= 0) break;;
        }

        if(n <= 0)
        {
            return true;
        }
        return false;
    }
};


int main(int argc, char const *argv[])
{
    /* code */
    return 0;
}
