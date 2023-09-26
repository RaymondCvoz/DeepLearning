#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
using namespace std;

class LRUCache
{
public:
    struct Node
    {
        Node *prev;
        Node *next;
        int val;
        int num;
        Node()
        {
            prev = nullptr;
            next = nullptr;
            val = 0;
        }
    };

    int total_capacity, lens;

    map<int, Node *> pool;

    Node *front = nullptr;
    Node *rear = nullptr;

    LRUCache(int capacity)
    {
        this->total_capacity = capacity;

        front = new Node();
        rear = new Node();

        front->next = rear;
        rear->prev = front;
    }

    int get(int key)
    {
        if (pool[key] == nullptr)
        {
            return -1;
        }
        else
        {
            int val = pool[key]->val;
            move_to_front(key, val);
        }
        return pool[key]->val;
    }

    void put(int key, int value)
    {

        if (pool[key] == nullptr)
        {
            if (lens == total_capacity)
            {
                delete_from_list(rear->prev);
                Node *cnt = new Node();
                cnt->val = value;
                cnt->num = key;
                add_to_list(cnt);
                pool[key] = cnt;
            }
            else
            {
                lens++;
                Node *cnt = new Node();
                cnt->val = value;
                cnt->num = key;
                add_to_list(cnt);
                pool[key] = cnt;
            }
        }
        else
        {
            move_to_front(key, value);
        }
    }

    void add_to_list(Node *cnt)
    {
        front->next->prev = cnt;
        cnt->next = front->next;
        cnt->prev = front;
        front->next = cnt;
    }

    void delete_from_list(Node *cnt)
    {
        cnt->prev->next = cnt->next;
        cnt->next->prev = cnt->prev;
        pool[cnt->num] = nullptr;
        // free(cnt);
    }

    void move_to_front(int key, int val)
    {
        Node *tmp = new Node();

        Node *cnt = pool[key];

        tmp->val = val;
        tmp->num = key;
        add_to_list(tmp);
        pool[key] = tmp;

        cnt->prev->next = cnt->next;
        cnt->next->prev = cnt->prev;
        // free(cnt);
    }
};

/**
 * Your LockingTree object will be instantiated and called as such:
 * LockingTree* obj = new LockingTree(parent);
 * bool param_1 = obj->lock(num,user);
 * bool param_2 = obj->unlock(num,user);
 * bool param_3 = obj->upgrade(num,user);
 */
int main(int argc, char const *argv[])
{
    /* code */
    return 0;
}