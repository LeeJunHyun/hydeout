---
layout: post
title: '(작성중) Test LaTex using MathJax'
categories:
  - Coding
tags:
  - web
  - html
  - tutorial
---

<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}	  
    });
</script>
<script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

<p>$ x $에 관한 이차방정식 $ ax^2 + bx + c = 0 $의 해는 다음과 같다.</p>
<p>$$ x = \frac{-b \pm \sqrt{b^2-4ac}}{2a} $$</p>