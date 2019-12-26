### <span id = 'id2'>2. 排序算法</span>

#### <span id = 'id21'>2.1 冒泡排序</span>

```python
def bubble_sort(nums):
    n = len(nums)
    if n < 2:
        return
    for i in range(n - 1):
        is_sorted = True
        for j in range(n - i - 1):
            if nums[j + 1] < nums[j]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                is_sorted = False
        if is_sorted:
            break
```

**复杂度分析：**当数组一开始就有序时，此时的时间复杂度最低为$O(n)$，即只需要遍历一次即可；当数组初始为逆序时，此时的时间复杂度最高为$O(n^2)$；所以冒泡排序的平均时间复杂度为$O(n^2)$，空间复杂度为$O(1)$。

#### <span id = 'id22'>2.2 选择排序</span>

```python
def select_sort(nums):
    n = len(nums)
    if n < 2:
        return
    for i in range(n-1):
        mid_loc = i
        for j in range(i+1,n):
            if nums[mid_loc] > nums[j]:
                mid_loc = j
        nums[mid_loc],nums[i] = nums[i],nums[mid_loc]
```

**复杂度分析：**由于选择排序算法每一轮都是要找当前最小（或最大）的值，因此都要挨个的进行比较，所以选择排序算法的时间复杂度在各种情况下均为$O(n^2)$，空间复杂度为$O(1)$。

#### <span id = 'id23'>2.3 插入排序</span>

```python
def insert_sort(nums):
    n = len(nums)
    if n < 2:
        return
    for i in range(n-1):
        for j in range(i,-1,-1):
            if nums[j+1] < nums[j]:
                nums[j],nums[j+1] = nums[j+1],nums[j]
            else:
                break
```

**复杂度分析：**当数组一开始就有序时，此时的时间复杂度最低为$O(n)$，即只需要遍历一次即可；当数组初始为逆序时，此时的时间复杂度最高为$O(n^2)$；所以插入排序算法的时间复杂度为$O(n^2)$，空间复杂度为$O(1)$。

#### <span id = 'id24'>2.4 快速排序</span>

**方法一：**

```python
def quick_sort(nums):
    n = len(nums)
    if n < 2:
        return nums
    mid = nums[n // 2]
    left, right = [], []
    nums.remove(mid)
    for item in nums:
        if item > mid:
            right.append(item)
        else:
            left.append(item)
    return quick_sort(left) + [mid] + quick_sort(right)
```

**复杂度分析：**时间复杂度为$O(n\log{n})$，空间复杂度$O(n)$。虽然这种实现代码简单，但空间复杂度却不是最优。

**方法二：**

**思路**：结题思路同剑指Offer上的一样，分为两步：①选择一个哨位，然后将数组分成左右两个部分，左边均小于哨位，右边均大于哨位；②递归划分左右两个部分；

```python
def Partion(nums, start, end):
    index = (end - start + 1) // 2 + start # 选择哨位，这里每个都选择中间这个
    nums[index], nums[end] = nums[end], nums[index]
    small = start - 1 # 用来指向左边小于末尾元素的最后一个
    for index in range(start, end):
        if nums[index] < nums[end]:
            small += 1
            if small != index:
                nums[small], nums[index] = nums[index], nums[small]
    small += 1
    nums[small], nums[end] = nums[end], nums[small]
    return small


def quick_sort(nums, start, end):
    if start == end:
        return
    index = Partion(nums, start, end)
    print(index, nums)
    if index > start:
        quick_sort(nums, start, index - 1)
    if index < end:
        quick_sort(nums, index + 1, end)


if __name__ == '__main__':
    s = [4, 6, 3, 2, 0, 1, 8]
    quick_sort(s, 0, len(s) - 1)
    print(s)
```

**复杂度分析**：如果待排序数组一开始就是有序的，且如果每次有选择最后一个作为哨位；此种情况下，第一轮需要将前$n-1$个数同哨位比较，由于此时的哨位为最大值，所有划分后的右边部分为空，左边部分剩下$n-1$个数；接着继续对左边的部分进行递归划分，由于数组有序且仍是最后一个为哨位，此次划分就需要进行$n-2$次比较，划分后的情况同前一次一样，因此这种情况下的时间复杂度为$(n-1)+(n-2)+\cdots +1=O(n^2)$。如果数组一开始就是有序的，且每次都选择中间位置的数作为哨位，或者数组的乱序的，但是运气好每次选择的哨位都是中位数，那么总共需要比较的次数如下：

第一轮：比较$n-1$次，然后数组被划分为左右两个部分，标记为left,right；

第二轮：left部分比较$\frac{n-1}{2}-1=\frac{n-3}{2}$次，然后划分为两个部分，标记为left_left，left_right

​				right部分同样比较$\frac{n-3}{2}$，然后划分为两个部分，标记为right_left，right_right

​				一种比较$(n-3)$次

第三轮：left_left部分比较：$\frac{n-3}{4}-1=\frac{n-7}{4}$，其余部分同样，总共比较$(n-7)$次

第$\log{n}$轮：也就是最后一轮，总共比较$(n-(2^{\log{n}}-1))$次

因此，总的时间复杂度为：
$$
\begin{aligned}
O(T)&=(n-1)+(n-3)+\cdots+(n-(2^{\log{n}}-1))\\[2ex]
&=n\cdot\log{n}-(1+3+7+\cdots+2^{\log{n}}-1)\\[2ex]
&=n\cdot\log{n}-\left[(2^1-1)+(2^2-1)+\cdots+(2^{\log{n}}-1)\right]\\[2ex]
&=n\cdot\log{n}-\left[\frac{2\cdot(1-2^{\log{n}})}{1-2}-\log{n}\right]\\[2ex]
&=n\cdot\log{n}+\log(n)+2-2\cdot n
\end{aligned}
$$
所以时间复杂度最好为$O(n\cdot\log{n})$，最差为$O(n^2)$，平均为$O(n\cdot\log{n})$，空间复杂度同最好情况下为递归的深度$O(\log n)$。

#### [返回目录](./README.md)

### <span id = 'id3'>3. 数组中重复的数字</span>

该题目在LeetCode上最相近的题目为：leetcode 287，也就是寻找重复数。

#### <span id = 'id31'>3.1 [寻找重复数 No.287（中等） ](https://leetcode-cn.com/problems/find-the-duplicate-number/)</span>

> 给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。

**方法一：**

思路：用字典来保存每一个不同数字所出现的数字，当出现次数大于1时，则说明该数字重复。

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        dicts = {}
        for item in nums:
            dicts[item] = dicts.get(item,0)+1
            if dicts[item] > 1:
                return item         
```

<font color  = blue >时间复杂度 $O(n)$，空间复杂度$O(n)$</font>

**方法二：**

采用原书上的思路，不过要略微做一点调整。

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        nums = [0] + nums
        for i in range(1,len(nums)):
            while i != nums[i]:
                if nums[i] == nums[nums[i]]:
                    return nums[i]
                else:
                    tmp = nums[i]
                    nums[i] = nums[tmp]
                    nums[tmp] = tmp
```

<font color = blue>时间复杂度$O(n)$，空间复杂度$O(1)$</font>

**方法三：**

第三种做法实际上就剑指Offer中面试题三中题目二，其要求是不能修改数组，也就是采用方法二的思路行不通了。

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        start,end = 1,len(nums)-1 # 
        count_l,count_r = 0,0
        while start <= end:
            middle = (start+end)//2 
            if middle == start:
                break
            for num in nums:
                if start <= num and num <= middle:
                    count_l += 1
                elif middle < num and num <= end:
                    count_r += 1
            if count_l > (middle - start + 1):
                end = middle
            else:
                start = middle+1
            count_l = 0
            count_r = 0

        count = 0
        for num in nums:# 从两个确定有重复数之中找其中一个
            if num == start:
                count += 1
            if count > 1:
                return start
        return end
```

<font color = blue>时间复杂度$O(\log{n})\cdot O(n)=O(n\log{n})$，空间复杂度$O(1)$</font>

#### <span id = 'id32'>3.2 [存在重复元素 No.217（简单）](https://leetcode-cn.com/problems/contains-duplicate/)</span>

这个题目与287的区别在于，该题目中没有再限制原素中每个值的范围，也就是说可以为负整数，或者大于n。

**方法一：**

直接利用`set`来完成。

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums) != len(set(nums))
```

**方法二：**

与3.1方法一相同，该方法可以算得上是找重复数的一个万能方法了，不管是最后要返回重复的值，或者是索引都能按照这个模板来（见3.3）。

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        dicts = {}
        for item in nums:
            dicts[item] = dicts.get(item,0)+1
            if dicts[item] > 1:
                return True
        return False
```

<font color = blue>时间复杂度 $O(n)$，空间复杂度$O(n)$</font>

#### <span id = 'id33'>3.3 [存在重复元素 II No.219（简单）](https://leetcode-cn.com/problems/contains-duplicate-ii/)</span>

> 给定一个整数数组和一个整数 k，判断数组中是否存在两个不同的索引 i 和 j，使得 nums [i] = nums [j]，并且 i 和 j 的差的绝对值最大为 k。
>

**方法一：**

该方法本质上还是上面说的万金油方法，通过一个字典来保存每个元素出现的索引；当出现重新数字时判断是否符合条件，符合则返回，不符合则去掉第一次出现的索引。

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        dicts = {}
        for idx,item in enumerate(nums):
            dicts[item] = dicts.get(item,[])+[idx] # 用字典来保存每个元素的索引
            if len(dicts[item]) == 2:
                if dicts[item][1] - dicts[item][0]<= k:
                    return True
                else:
                    dicts[item].pop(0)
        return False
```

<font color = blue>时间复杂度为$O(n)$，空间复杂度为$O(n)$ </font>

#### [返回目录](./README.md)



### <span id = 'id4'>4. 二维数组中的查找</span>

该题目在LeetCode上最相近的题目为：leetcode 74，也就是搜索二维矩阵。

#### <span id = 'id41'>4.1 [搜索二维矩阵 No.74（中等）](https://leetcode-cn.com/problems/search-a-2d-matrix/)</span>

> 编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
>
> 每行中的整数从左到右按升序排列。
> 每行的第一个整数大于前一行的最后一个整数。

**方法一：**

解题思路同剑指Offer上的一致，先定位区域，再依次遍历。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        n,m = len(matrix),len(matrix[0])
        row,col = n-1,m-1
        for i in range(n):
            if matrix[i][0] >= target:
                row = i
                break
        for i in range(m):
            if matrix[0][i] >= target:
                col = i
                break
        for i in range(row+1):
            for j in range(col+1):
                if target == matrix[i][j]:
                    return True
        return False
```

<font color = blue>当目标值为二维数组最后一个元素时情况最差，所以平均时间复杂度为$O(n^2)$，空间复杂度为$O(1)$</font>

#### <span id = 'id42'>4.2 [搜索二维矩阵 II No.240（中等）](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)</span>

> 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：
>
> 每行的元素从左到右升序排列。
> 每列的元素从上到下升序排列。

这个题目和4.1是一样的解法，直接把4.1的代码复制过去就能通过。

#### [返回目录](./README.md)



### <span id = 'id5'>5. 替换空格</span>

该题目在LeetCode上还没找到对应题目。

**方法一：**

解题思路同剑指Offer上的一致，先计算替换后的总长度，然后从后向前开始遍历并移动。同时，下面的代码实现了一个通用的替换，不仅仅局限于3个字符。

```python
def replaceBlank(sentence, r='20%'):
    if len(sentence) < 1:
        return
    num_blank = 0
    index_of_original = len(sentence) - 1
    s = []
    for i in range(len(sentence)):
        if sentence[i] == ' ':
            num_blank += 1
        s.append(sentence[i])
        
    s += [' '] * (len(r) - 1) * num_blank
    
    index_of_new = len(s) - 1
    while index_of_original > -1:
        if s[index_of_original] == ' ':
            for i in range(len(r) - 1, -1, -1):
                s[index_of_new] = r[i]
                index_of_new -= 1
        else:
            s[index_of_new] = s[index_of_original]
            index_of_new -= 1
        index_of_original -= 1
    return "".join(s)
```

<font color = blue>时间复杂度为$O(n*m)$，其中$m$是这个替换串的长度；空间复杂度在这个代码中是$O(n+m*num\_blank)$</font>

### <span id = 'id6'>6. 从头到尾打印链表</span>

该题目在LeetCode上最相近的题目为：leetcode 206，也就是反转链表。

#### <span id = 'id61'>6.1 [反转链表 No.206（简单）](https://leetcode-cn.com/problems/reverse-linked-list/)</span>

> 反转一个单链表
>
> 输入: 1->2->3->4->5->NULL
> 输出: 5->4->3->2->1->NULL

对于这个题目，一把都是通过递归来处理，但是根据题目的不同要求（也就是能不能改变链表的结构），可以有两种方式来递归。

**方法一：**

```python
# 要求返回的是一个链表

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        new_list = self.reverseList(head.next)
        t = head.next
        t.next = head
        head.next = None
        return new_list
```

<font color = blue>时间复杂度为$O(n)$，空间复杂度为$O(n)$</font>

**方法二：**

```python
# 直接输出结果
def reverse_list(h):
    if not h:
        return 
    if h.next:
        reverse_list(h.next)
    print(h.val)
```

<font color = blue>时间复杂度为$O(n)$，空间复杂度为$O(1)$</font>

#### <span id = 'id62'>6.2 [反转链表 II No.92（中等）](https://leetcode-cn.com/problems/reverse-linked-list-ii/)</span>

> 反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
>
> 说明:
> 1 ≤ m ≤ n ≤ 链表长度。
>
> 示例:
>
> 输入: 1->2->3->4->5->NULL, m = 2, n = 4
> 输出: 1->4->3->2->5->NULL

**方法一：**

做这道题目的主要思路是：先在原始链表最前面添加一个节点方便处理一些特殊情况（m=1）时；然后将需要反转的节点放到一个列表中，再遍历一遍列表进行反转；最后将对应处链接好即可。

```python
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        if not head or not head.next or m == n:
            return head

        node = ListNode(0)
        node.next = head
        p,q = node,node
        pools,idx  = [],0

        while len(pools) < n-m+ 1:# 加入列表
            if idx < m - 1:
                p = p.next
                q = p
                idx += 1
                continue
            pools.append(q.next)
            q = q.next
        
        q = q.next
        for i in range(len(pools)-1,0,-1):# 反转
            pools[i].next = pools[i-1]

        pools[0].next = q # 链接
        p.next = pools[-1]
        return node.next
```

<font color = blue>时间复杂度最坏为全部反转，因此平均时间复杂度为$O(n)$，空间复杂度为$O(n-m)$</font>

#### [返回目录](./README.md)



### <span id = "id7">7. 重建二叉树</span>

该题目在LeetCode上最相近的题目为：leetcode 105，也就是从前序与中序遍历构造二叉树。

#### <span id = 'id71'>7.1 [从前序与中序遍历序列构造二叉树 No.105（中等）](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)</span>

**方法一：**

按照根据前序和中序手动恢复二叉树的思路来递归恢复。

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if len(preorder) == 0 and len(inorder) == 0:
            return None
        root = TreeNode(preorder[0])# 找到根节点
        idx = inorder.index(preorder[0])# 划分左右子树
        root.left = self.buildTree(preorder[1:idx+1],inorder[:idx])# 递归构造左右子树
        root.right = self.buildTree(preorder[idx+1:],inorder[idx+1:])
        return root
```

<font color = blue>时间复杂度为$O(n)$，空间复杂度也为$O(n)$，$n$表示节点数。</font>

#### <span id = 'id72'>7.2 [从中序与后序遍历序列构造二叉树 No. 106（中等）](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)</span>

**方法一：**

按照根据中序和后序手动恢复二叉树的思路来递归恢复。

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if len(inorder) == 0 and len(postorder) == 0:
            return None
        root = TreeNode(postorder[-1])
        idx = inorder.index(postorder[-1])
        root.left = self.buildTree(inorder[:idx],postorder[:idx])
        root.right = self.buildTree(inorder[idx+1:],postorder[idx:-1])
        return root
```

#### [返回目录](./README.md)



### <span id = "id8">8. 二叉树的下一个节点</span>

该题目在LeetCode上还没找到对应题目，可以用[牛客网上的来代替](https://www.nowcoder.com/questionTerminal/9023a0c988684a53960365b889ceaf5e)。

**方法一：**

结题方案和书上的一样，分为三种情况：①有右子树；②没有右子树但父节点的左节点是它；③算是②的一个扩展，所以代码里面就分了两种情况；

```python
class Solution:
    def GetNext(self, pNode):
        # write code here
        if not pNode:
            return pNode
        if pNode.right:# 如果有右子树
            p = pNode.right
            while p.left:
                p = p.left
            return p
        else:
            while pNode.next:
                if pNode.next.left == pNode:
                    return pNode.next
                pNode = pNode.next
            return None
```

<font color = blue>时间复杂度最坏情况下为$O(\log{n})$，即遍历整个树深度的情况，所以平均时间复杂度为$O(\log{n})$；空间复杂度为$O(1)$。</font>

### <span id = "id9">9. 用两个栈实现队列</span>

该题目在LeetCode上最相近的题目为：leetcode 232，也就是用栈实现队列。

#### <span id = "id91">9.1 [ 用栈实现队列 No.232（简单）](https://leetcode-cn.com/problems/implement-queue-using-stacks/)</span>

队列的特点是先进先出，而栈的特点是先进后出，所以用两个栈来实现队列的思路，就是在出队的时候通过两个栈来进行”倒腾“（同剑指Offer上一致）。

**方法一：**

```python
class MyQueue:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.s1 = []
        self.s2 = []
    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        while self.s2:# 如果s2不为空
            t = self.s2.pop()
            self.s1.append(t)
        self.s1.append(x)
    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        while self.s1:# 如果s1不为空
            t = self.s1.pop()
            self.s2.append(t)
        return self.s2.pop()
    def peek(self) -> int:
        """
        Get the front element.
        """
        while self.s1:# 如果s1不为空
            t = self.s1.pop()
            self.s2.append(t)
        return self.s2[-1]
    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return len(self.s1) ==0 and len(self.s2) ==0
```

#### <span id = "id92">9.2 [用队列实现栈 No.225（简单）](https://leetcode-cn.com/problems/implement-stack-using-queues/)</span>

用队列来实现栈，这时候就不要用两个队列来实现了，只需要一个队列外加一个长度就可以。其思路就是，入栈的时候直接往队列里面添加就行，但是出栈的时候依次将队列里的前n-1个元素出队，依次加入到队列末尾就行，此时队列的队首就是栈顶元素。

**方法一：**

```python
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.quene = []
        self.length = 0  
    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.quene.append(x)
        self.length += 1  
    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        for _ in range(self.length - 1):
            t = self.quene.pop(0)
            self.quene.append(t)
        self.length -= 1
        return self.quene.pop(0)
    def top(self) -> int:
        """
        Get the top element.
        """
        t = self.pop()
        self.push(t)
        return t
    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return self.length < 1
```

#### [返回目录](./README.md)

### <span id = "id10">10. 斐波那契额数列</span>

该题目在LeetCode上对应的题目为：leetcode 509。

#### <span id = "id101">10.1 [斐波那契数 No.509（简单）](https://leetcode-cn.com/problems/fibonacci-number/)</span>

对于求解斐波那契数一般来说有两种，第一种就是自上而下的递归，第二种就是自下而上的动态规划。

**方法一：**

通过递归来求解。

```python
class Solution:
    def fib(self, N: int) -> int:
        if N == 0:
            return 0
        if N == 1:
            return 1
        return self.fib(N-1) + self.fib(N-2)
```

**复杂度分析：**可以看出，其时间复杂度和空间复杂度均主要花费在递归调用上，因此其空间复杂度为$O(n)$；而由于才用递归的方法会反复计算同一个数的斐波那契数多次，故其时间复杂度高达$2^n-1=O(2^n)$。

**方法二：**

通过动态规划来求解，也就是从f[2]开始，按公式一直计算到f[n]。

```python
class Solution:
    def fib(self, N: int) -> int:
        # 状态转移方程：f[i] = f[i-1] + f[i-2]，其中f[i]表示第i个数斐波那契数
        if N < 2:
            return 
        f = (N+1) *[0]
        f[0],f[1] = 0,1
        for i in range(2,N+1):
            f[i] = f[i-1]+f[i-2]
        return f[-1]
```

**复杂度分析：**相比较来说，使用动态规划的话其空间复杂度主要是花费在保存历史结果上，而时间复杂度也是花费在每个斐波那契数的计算上，且每个数都只计算了一次，因此其时间复杂度和空间复杂度均为$O(n)$。

#### <span id = "id102">10.2 [爬楼梯 No.70（简单）](https://leetcode-cn.com/problems/climbing-stairs/)</span>

爬楼梯这个题对应的也就是青蛙跳台阶的问题，和求解斐波那契数一样，同样也可以按照递归和动态规划来求解。

**方法一：**

通过自上而下递归来求解。假设下载有n个台阶需要你跳上去，那么当你跳最后一次的时候，你可能是通过一步跳上去的，同时你也可能是通过跳两步跳上去的，因此f[n] = f[n-1] + f[n-2]。

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        return self.climbStairs(n-1) + self.climbStairs(n-2)
```

**复杂度分析：**其时间和空间复杂度的分析和结果均10.1方法一，所以就不再赘述。遗憾的是虽然代码正确，但由于其时间复杂度太高，这个代码在leetcode上自然是无法通过的，超时。所以也就只能用下面一种方法。

**方法二：**

通过自下而上的动态规划来求解。也就是你要算n阶楼梯有多少种可能，那我就从有3阶开始算，一直重下往上根据公式算到第n阶。

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        f = [0] *(n+1)
        f[1],f[2] = 1,2
        for i in range(3,n+1):
            f[i] = f[i-1] + f[i-2]
        return f[-1]
```

**复杂度分析：**其时间和空间复杂度的分析和结果均同上10.1方法二，所以就不再赘述。

#### <span id = "id103">10.3 [跳跃游戏 No.55（中等）](https://leetcode-cn.com/problems/jump-game/)</span>

> 给定一个非负整数数组，你最初位于数组的第一个位置。
>
> 数组中的每个元素代表你在该位置可以跳跃的最大长度。
>
> 判断你是否能够到达最后一个位置。
>
> ```
> 输入: [2,3,1,1,4]
> 输出: true
> 解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。
> ```

这个题目也是一个关于跳的题，不过它计算的是能不能跳到最后一个台阶。

其解题的思路是用一个变量maxs来表示当前能跳跃的最大台阶数，每跳一次减一，然后和下一个石头上的数两者取最大的一个，如果能跳到倒数第二个石头，且maxs不为零，那么就表示能够跳到最后一个石头。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        maxs = 0
        for i in range(len(nums)-1):
            maxs -= 1
            maxs = max(maxs,nums[i])
            if maxs == 0:
                return False
        return True
```

**复杂度分析：**由于代码并不复杂，所以一眼可以看出其时间复杂度为$O(n)$，空间复杂度为$O(1)$

#### [返回目录](./README.md)

### <span id = "id11">11. 旋转数组的最小值</span>

该题目在LeetCode上对应的题目为：leetcode 153。

#### <span id = "id111">11.1 [寻找旋转排序数组中的最小值 No.153（中等）](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)</span>

这个题的解题思路同剑指Offer上的一致，采用的是二分查找，其关键就是判断最小值位于前后哪个部分的问题。

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return nums[0]
        left,right = 0,n-1
        mid_idx = left
        while nums[left] >= nums[right]:
            if right - left == 1:
                mid_idx = right
                break
            mid_idx = (left + right) // 2
            if nums[mid_idx] >= nums[left]:# 如果中间值大>=左边值，则前面肯定递增，最小值在后
                left = mid_idx
            elif nums[mid_idx] <= nums[right]:#如果中间值<=右边值，则后面有序，最小值在前边
                right = mid_idx
        return nums[mid_idx]
```

上面这段代码没有考虑书上列举的[1,0,1,1,1]这种情况也能通过，一开始以为是国内版leetcode-cn.com的bug，切换国外leetcode.com也能通过。看来要么是leetcode上的递增指的是严格递增，要么就是bug了。完整如下：

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        def minInOrder(nums, left, right):
            min_idx = left
            for i in range(left, right + 1):
                if nums[i] < nums[min_idx]:
                    min_idx = i
            return min_idx   

        n = len(nums)
        if n < 2:
            return nums[0]
        left,right = 0,n-1
        mid_idx = left
        while nums[left] >= nums[right]:
            if right - left == 1:
                mid_idx = right
                break
            mid_idx = (left + right) // 2
            if nums[left] == nums[mid_idx] == nums[right]:
                mid_idx = minInOrder(nums, left, right)
                break
            elif nums[mid_idx] >= nums[left]:
                left = mid_idx
            elif nums[mid_idx] <= nums[right]:
                right = mid_idx
        return nums[mid_idx]
```

**复杂度分析：**可以看出，最好情况顺序时的时间复杂度为$O(1)$，最差情况下如[1,0,1,1,1]时为$o(n)$，其次一般情况下为$O(\log{n})$，则平均时间复杂度为$O(\log{n})$吗？（笔者还真不确定）。空间复杂度为$O(1)$。

#### <span id = "id112">11.2 [寻找旋转排序数组中的最小值 II No.154（困难）](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)</span>

> 假设按照升序排序的数组在预先未知的某个点上进行了旋转。
>
> ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
>
> 请找出其中最小的元素。
>
> 注意数组中可能存在重复的元素，如：
>
> ```
> 输入: [2,2,2,0,1]
> 输出: 0
> ```

当看完这个题目的要求，突然就有点”踏破铁鞋无觅处，得来全不费工夫的感觉“。这就是第153题的加强版，也就是考虑了剑指Offer上的特殊情况，不再赘述。下面再来看一道类似思路题目。

#### <span id = "id113">11.3 [搜索旋转排序数组 No.33（中等）](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)</span>

> 假设按照升序排序的数组在预先未知的某个点上进行了旋转。搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。你可以假设数组中不存在重复的元素。
>
> 示例 1:
>
> 输入: nums = [4,5,6,7,0,1,2], target = 0
> 输出: 4
> 示例 2:
>
> 输入: nums = [4,5,6,7,0,1,2], target = 3
> 输出: -1

可以看到，题目中特意提到了不存在重复元素。解决这个题是思路也会用到上面的想法，根据中间值与两边值的关系判断出哪一边是有序，然后再判断目标值属于那一段中，然后再迭代。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n < 1:
            return -1
        if n == 1:
            if nums[0] == target:
                return 0
            else:
                return -1
        left,right = 0,n-1
        while left < right :
            idx = (left + right) // 2
            if nums[idx] == target:
                return idx
            if nums[idx] >= nums[left]: # 说明左边有序
                if nums[left] <= target and target < nums[idx]: # 且target在左边
                    right = idx
                else:
                    left = idx
            elif nums[idx] <= nums[right]: #说明右边有序
                if nums[idx] <= target and target <= nums[right]:# 且target在右边
                    left = idx
                else:
                    right = idx
            if right - left == 1:
                if nums[left] == target:
                    return left
                if nums[right] == target:
                    return right
                else:
                    return - 1
```

**复杂度分析：**时间复杂度为$O(\log{n})$，空间复杂度为$O(1)$。

#### <span id = "id114">11.4 [搜索旋转排序数组 II No.81（中等）](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)</span>

> 假设按照升序排序的数组在预先未知的某个点上进行了旋转。
>
> ( 例如，数组 [0,0,1,2,2,5,6] 可能变为 [2,5,6,0,0,1,2] )。
>
> 编写一个函数来判断给定的目标值是否存在于数组中。若存在返回 true，否则返回 false。
>
> 示例 1:
>
> 输入: nums = [2,5,6,0,0,1,2], target = 0
> 输出: true

这个题同上一个题的区别就在于81题允许存在重复，因此也需要像154题一样加入一个顺序查找。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        def minInOrder(nums, left, right,target):
            for i in range(left, right + 1):
                if nums[i] == target:
                    return True
            return False

        n = len(nums)
        if n < 1:
            return False
        if n == 1:
            if nums[0] == target:
                return True
            else:
                return False
                
        left,right = 0,n-1
        while left < right:
            idx = (left + right) // 2
            if nums[idx] == target:
                return True
            if nums[left] == nums[idx] == nums[right]:
                return minInOrder(nums,left,right,target)
            if nums[idx] >= nums[left]:# 说明左边有序
                if nums[left] <= target and target < nums[idx]:# 且target在左边
                    right = idx
                else:
                    left = idx
            elif nums[idx] <= nums[right]:# 说明右边有序
                if nums[idx] < target and target <= nums[right]:# 且target在右边
                    left = idx
                else:
                    right = idx
            if right - left == 1:
                if nums[left] == target or nums[right] == target:
                    return True
                else:
                    return False

```

**复杂度分析：**时间复杂度为$O(\log{n})$，空间复杂度为$O(1)$

#### [返回目录](./README.md)

### <span id = "id12">12. 矩阵中的路径</span>

### <span id = "id13">13. 机器人的运动范围</span>

### <span id = "id14">14. 减绳子</span>

该题目在LeetCode上没有找到对应的题目为，可以用[牛客网](https://www.nowcoder.com/practice/57d85990ba5b440ab888fc72b0751bf8?tpId=13&tqId=33257&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)来代替。

这个题的结题思路同剑指Offer一样都是采用动态规划来求解。由于$f(n) = max(f(i)*f(n-i))$，而$f(i) = max(f(j)*f(i-j))$，即每个问题又被划分成更下的子问题，所以我们需要自下而上的进行求解。

```python
class Solution:
    def cutRope(self, number):
        # write code here
        # f(n) = max(f(1)*f(n-1),f(2)*f(n-2),...)
        if number == 2:
            return 1
        if number == 3:
            return 2
        f = [0] * (number+1)
        f[1],f[2],f[3] = 1,2,3
        for i in range(4,number+1):
            for j in range(1,i//2+1):
                f[i] = max(f[i],f[i-j]*f[j])
        return f[-1]
```

**复杂度分析：**其循环的执行次数为$\frac{1}{4}(n^2-1)$，因此其时间复杂度为$O(n^2)$；空间复杂度为$O(n)$，即整个`f`列表的空间。

### <span id = "id15">15. 二进制中1的个数</span>

这个题在leetcode中对应的题目为：leetcode 191。

#### <span id = "id151">15.1 [位1的个数 No.191（简单）](https://leetcode-cn.com/problems/number-of-1-bits/)</span>

> 编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’ 的个数（也被称为汉明重量）。
>
> 示例 1：
>
> 输入：00000000000000000000000000001011
> 输出：3
> 解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。

这个题与书中的区别就是这儿直接输入的就是二进制形式。

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n&0xffffffff != 0:
            count += 1
            n = n & (n-1)
        return count
```

**复杂度分析：**从代码可以看出，时间复杂度为$O(count)$，即$O(1)$，空间复杂度也为$O(1)$

### <span id = "id16">16. 数值的整数次方</span>

该题目在leetcode中对应的题目为：leetcode 50.

#### <span id = "id161">16.1 [Pow(x, n) No.50（中等）](https://leetcode-cn.com/problems/powx-n/)</span>

> 实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数。
>
> **示例 1:**
>
> ```python
> 输入: 2.00000, 10
> 输出: 1024.00000
> ```

解题思路了完全同剑指Offer一致。

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        
        def power(base,exp):
            if exp == 0:
                return 1
            if exp == 1:
                return base
            result = power(base,exp>>1)
            result *= result
            if exp & 0x1 == 1: # 判断exp是否为奇数
                result *= base
            return result
        
        if x == 0 or x == 1:
            return x
        if n == 0:
            return 1
        sign = True
        if n < 0:
            sign = False
            n *= -1
        result = power(x,n)
        return result if sign > 0 else 1/result
```

**复杂度分析：** 时间复杂度为$O(\log{n})$，空间复杂度也为$O(\log{n})$，因为每递归一次`power`都会开辟$O(1)$的空间。

#### [返回目录](./README.md)

### <span id = "id17">17. 打印从1到最大的n位数</span>

### <span id = "id18">18. 删除链表的节点</span>

这个题在leetcode中完全对应的题目应该没有，只有一个类似的也就是leetcode 237。与原书中唯一的差别就是书中给出了头节点，而237中没有给出。

#### <span id = "id181">18.1 [删除链表中的节点 No.237（简单）](https://leetcode-cn.com/problems/delete-node-in-a-linked-list/)</span>

> 请编写一个函数，使其可以删除某个链表中给定的（非末尾）节点，你将只被给定要求被删除的节点。
>
> 示例 1:
>
> 输入: head = [4,5,1,9], node = 5
> 输出: [4,1,9]
> 解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.

解题方法同书上一致，将下一个节点的值赋值到当前要删除的节点，然后将当前节点链接到下个节点所指向的地址。

```python
class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next
```

看起要比书上的情况简单很多，如果是按照书上的情况，应该如下：

```python
def deleteNode(head,node):
    if node.next:# 删除节点不是尾节点
        node.val = node.next.val
        tmp = node.next
        node.next = node.next.next
        del tmp
    elif head == node:# 删除节点是头节点且只有一个节点
        tmp = head
        head = None
        del tmp
    else:#删除为尾节点
        p = head
        while p.next != node:
            p = p.next
        p.next = None
        del node
```

**复杂度分析：**时间复杂度最好情况下为$O(1)$，最坏情况下为$O(n)$，因此平均时间复杂度为$[(n-1)\cdot O(1)+O(n)]/n=O(1)$，空间复杂度为$O(1)$

#### <span id = "id182">18.2 [删除排序链表中的重复元素 No.82（简单）](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)</span>

> 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
>
> **示例 1:**
>
> ```python
> 输入: 1->1->2
> 输出: 1->2
> ```

由于是已经排序好的，所以只需要一次遍历即可去掉重复的部分。

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        post,pre = head,head.next
        while pre:
            if pre.val == post.val:
                pre = pre.next
            else:
                post.next = pre
                post = pre
                pre = pre.next
        post.next = pre
        return head 
```

**复杂度分析：**时间复杂度为$O(n)$，空间复杂度为$O(1)$



#### [返回目录](./README.md)















