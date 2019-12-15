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

**复杂度分析：**时间复杂度为$O(n\log{n})$，空间复杂度$O(n\log{n})$。虽然这种实现代码简单，但空间复杂度却不是最优。

**方法二：**



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















