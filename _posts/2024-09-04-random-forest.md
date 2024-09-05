---
title: Rừng Cây (Random Forest)
author: Andrew Benedict
date: 2024-09-03
category: Jekyll
layout: post
---

> **Lưu ý**
>
> Một số thuật ngữ liên quan đến thuật toán trong bài viết này sẽ không được dịch ra tiếng Việt
{:.block-warning}

# 1. Khái niệm về Random Forest
Giả sử trong tình huống bạn đang muốn mua một chiếc ô tô cho bản thân nhưng lại không biết lựa chọn hãng xe nào là hợp lí nhất. Vì chẳng biết rõ được mình nên chọn chiếc xe nào, bạn hỏi mọi người xung quanh mong chờ được tư vấn. Có rất nhiều đáp án mà họ có thể gợi ý, nhưng bạn để ý rằng là 7 trên 10 người trả lời rằng chiếc Honda Civic 2019 là phù hợp với bạn nhất nên bạn quyết cọc tiền mua chiếc xe ngay trong hôm đó. Nếu bạn để ý, hiện tượng thu thập ý kiến từ nhiều nguồn để đưa ra được quyết định cuối cùng qua số đông (majority voting) xảy ra rất thường xuyên trong cuộc sống; thậm chí nó còn có tên riêng luôn là *trí thông minh đám đông* (wisdom of the crowd). Thuật toán Random Forest (RF) chính xác là hoạt động dựa trên ý tưởng này, nhưng trong Machine Learning, nó được biến đến với thuật ngữ *bagging* thuộc một mảng to hơn là *ensemble*. Ensemble là kĩ thuật để gia tăng hiệu năng hay độ chính xác của thuật toán cũng như giúp model ít bị bias hay high variance (hay còn gọi là overfit). Mảng này chứa một số kĩ thuật khác, nhưng bài này sẽ chỉ nói về nhánh bagging của RF. <br>

![definition example](../images/random_forest/example.png)

Như tên gọi, RF là tập hợp nhiều cây quyết định và đưa ra kết quả theo số đông của các cây đó. Ví dụ trong bài toán dự đoán mail spam, thì nếu một model RF gồm 50 cây quyết định, trong đó có 40 cây cho kết quả là "spam" và 10 cây cho kết quả ngược lại, thì model sẽ dự đoán là "spam". Cách hoạt động thì đơn giản như vậy, nhưng RF còn có một số kĩ thuật khác để tăng hiệu quả của model cũng như "khỏe" (robust) hơn.

# Tài liệu tham khảo
[https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/)