---
layout: post
title: '[Web] Jekyll을 이용해서 개인 Github Page 만들기 - 부가기능'
categories:
  - Coding
tags:
  - web
  - ruby
  - jekyll
  - html
  - tutorial
---

지난 포스트 [[Web] Jekyll을 이용해서 개인 Github Page 만들기](https://leejunhyun.github.io/coding/2018/09/01/Github-page.html) 에서는 Jekyll 테마를 받아 개인 페이지를 만드는 방법을 정리했습니다. 본 포스트에서는 여기에 부가기능들(댓글-disqus, google analytics, LaTex-MathJax, archive)을 넣는 방법을 정리할 것입니다.

---
---
우선 jekyll 테마를 받아서 사용한다면, 각자 어떤 부가기능들이 구현되어있나 살펴보는 것이 좋다. 보통은 간단하게 `_config.yml`만 수정하면 되게끔 만들어져있기 때문이다.


## Disqus 소셜 댓글 서비스
정적 사이트 생성기인 Jekyll과 github page 호스팅을 사용하여 개인 페이지를 만들면 댓글관리하기가 힘들다. 따로 데이터베이스에 댓글들을 정리해야하는데, 이때 주로 사용되는 서비스가 [disqus](https://disqus.com/)이다. 댓글 서비스를 직접 구현하지 않고, 위젯 형태로 서비스에 삽입할 수 있다. 다른 서비스들도 많지만, 기존 jekyll 테마들이 disqus들을 지원하는 경우가 많으므로 disqus를 다룬다.

우선 https://disqus.com/ 에 접속하여 회원가입을 한다. 보통은 가입할때 `Create a new site`를 하는데, 혹시 이미 가입되어있다면 본인 계정에 로그인을 하고, `admin`에 들어가서 사이트 추가를 하면 `Create a new site`를 할 수 있다.

![disqus](/assets/img/Coding/github-page-disqus.png)

이때 중간에 `Website Name`을 입력하는 란이 있는데, **여기 등록하는 이름을 잘 기억해야 한다.** 

![disqus](/assets/img/Coding/github-page-disqus-install.png)


생성을 완료한 후에는 `Install on a site`에서 맨 아래의 `Universal Embed Code`로 들어간다. `Jekyll` 플랫폼을 바로 선택해도 좋지만, 어차피 똑같다.

이때 나오는 코드는 보통 jekyll theme에 구현되어 있다. 내가 고른 [hydeout](https://github.com/fongandrew/hydeout) 테마에서는 `_includes/disqus.html` 에 아래와 같이 구현되어있다. 구현되어있지 않다면 추가하면 된다.

```html
{% raw %}
{% if site.disqus.shortname %}
  <div id="disqus_thread">
    <button class="disqus-load" onClick="loadDisqusComments()">
      Load Comments
    </button>
  </div>
  <script>
  /**
  *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW
  *  TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
  *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT:s
  *  https://disqus.com/admin/universalcode/#configuration-variables
  */
  var disqus_config = function () {
    this.page.url = "{{ page.url | absolute_url }}";
    this.page.identifier = "{{ page.guid or page.id }}" ||
                           "{{ page.url | absolute_url }}";
  }
  function loadDisqusComments() { // DON'T EDIT BELOW THIS LINE
    var d = document, s = d.createElement('script');
    s.src = '//{{ site.disqus.shortname }}.disqus.com/embed.js';
    s.setAttribute('data-timestamp', +new Date());
    (d.head || d.body).appendChild(s);
  }
  </script>
  <noscript>
    Please enable JavaScript to view the
    <a href="https://disqus.com/?ref_noscript">comments powered by Disqus</a>.
  </noscript>

{% elsif jekyll.environment != "production" %}
  <p>
    You are seeing this because your Disqus shortname is not properly set. To
    configure Disqus, you should edit your <code>_config.yml</code> to include
    either a <code>disqus.shortname</code> variable.
  </p>

  <p>
    If you do not wish to use Disqus, override the
    <code>comments.html</code> partial for this theme.
  </p>
{% endif %}
{% endraw %}
```

여기서는 `site.disqus.shortname`라는 변수로 사용자 설정을 가져오는데, 이 부분은 구현된 코드마다 `site.disqus.username` 등 다를수는 있다. 이름이 다르더라도 대체로 비슷한 역할을 하는 변수일 것이다. `site` 변수는 `_config.yml`에서 정의되는데, 아래와 추가하였다.

```yml
disqus:
  shortname: my-disqus-shortname
```
이때 `my-disqus-shortname`를 아까 disqus에서 `Website Name`로 등록했던 이름으로 바꿔주면 된다.

그리고 앞으로 게시할 글들 상단에 comments 속성을 추가한다. 테마에 따라서는 disqus shortname만 설정해주면 알아서 적용되는 것도 있다.

```yml
---
layout: default
comments: true
# other options
---
```

## Google Analytics
[Google Analytics](https://analytics.google.com/) 는 내 웹의 접속자들에 대한 여러가지 통계치를 알 수 있는 서비스다. [Google Analytics](https://analytics.google.com/)에 접속해서 `속성만들기`를 하면 웹 사이트를 추가 할 수 있다. 이때 `추적 ID가져오기` 를 누르면 disqus때처럼 웹에 삽입할 수 있는 코드가 나온다. 이것도 내 테마에서는 `_includes/google-analytics.html`에 아래와같이 구현되어있었다. 마찬가지로 없다면 추가하면 된다.

```html
{% raw %}
{% if jekyll.environment == 'production' and site.google_analytics %}
  <script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');
  ga('create', '{{ site.google_analytics }}', 'auto');
  ga('send', 'pageview');
  </script>
{% endif %}
{% endraw %}
```
여기서는 `site.google_analytics` 라는 변수명으로 내 설정을 가져온다. `_config.yml`에 `google_analytics`라는 변수에 `추적 ID가져오기`를 통해 조회했던 내 코드만 선언해주면 된다. 예를 들면 아래와 같다.

```yml
google_analytics: 'UA-125030638-1'
```

이후에는 서비스를 이용할 수 있다.


## LaTex
작성하는 포스트중에 수식을 넣는 경우가 있다. 이럴때 사용하는 문법이 LaTex이며, 이를 지원하는 서비스가 [MathJax](https://www.mathjax.org/)이다. 홈페이지에 들어가면 사용법과 livedemo까지 친절하게 지원해준다.

나는 아래 코드를 `_layouts/post.html`의 header에 넣어주었다.
```html
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML" async>
  </script>
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      
      tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
    });
  </script>
  <script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "AMS" } } }); </script>
```

이후에는 layout으로 post를 받아서 작성하는 글들에는 LaTex을 사용할 수 있다.


> <p>예시</p>
> <p>\(a \ne 0\) 에 관한 이차방정식  \(ax^2 + bx + c = 0\)의 해는 다음과 같다.</p>
> <p>$$ x = \frac{-b \pm \sqrt{b^2-4ac}}{2a} $$</p>



## Archive
게시글들을 모아두는 Archive는 구현해놓은 테마도 있고, 구현해놓지 않은 테마도 있다. 안타깝게도 내가 고른 테마는 구현해놓지 않았다. 최대한 라이브러리나 플러그인을 쓰려고 했지만, 뭔가 복잡하고 잘 안되었기 때문에 직접 추가해주었다. ~~그래서 조잡하다.~~

우선 sidebar에 Archive 를 넣고싶어서 `_includes/sidebar-icon-links.html`를 봤더니,

```html
{% raw %}
{% assign tags_page = false %}
{% assign search_page = false %}
{% for node in site.pages %}
{% if node.layout == "tags" %}
    {% assign tags_page = node %}
{% elsif node.layout == "search" %}
    {% assign search_page = node %}
{% endif %}
{% endfor %}
{% endraw %}
```
이렇게 사이드바에 태그검색과 게시물 검색기능을 `site.pages`에서 찾아서 추가하길래 일단 archive도 추가해놓았다.

```html
{% raw %}
{% assign tags_page = false %}
{% assign search_page = false %}
{% assign archive_page = false %}
{% for node in site.pages %}
{% if node.layout == "tags" %}
    {% assign tags_page = node %}
{% elsif node.layout == "search" %}
    {% assign search_page = node %}
{% elsif node.layout == "archive" %}
    {% assign archive_page = node %}
{% endif %}
{% endfor %}
{% endraw %}
```
그리고 태그검색기능이 어디서오나 봤더니 `root`에 `tags.html`이 있었다. 이 파일의 내용은
```yml
---
layout: tags
title: Tags
---
```
이것이 전부였고, 나도 따라서 내용이 아래가 전부인 `archive.html`을 `root`에 만들었다. 
```html
---
layout: archive
title: Archive
---
```
여기서 layout을 지정해주니까 `_layouts` 폴더에 가보니 아니나 다를까 `_layouts/tags.html`이 있었다. 내용은 아래에 나온것이 전부다.
```html
{% raw %}
---
layout: page
---

{% include tags-list.html %}
{% endraw %}
```
나도 따라서 `_layouts/archive.html`을 만들었다.
```html
{% raw %}
---
layout: page
---

{% include archive-list.html %}
{% endraw %}
```
이제 `_includes`로 가서 `_includes/archive-list.html`을 구현하면 된다.
```html
{% raw %}
{% assign postsByYear = site.posts | group_by_exp:"post", "post.date | date: '%Y'"  %}
{% for year in postsByYear %}
    {% assign postsByMonth = year.items | group_by_exp:"post", "post.date | date: '%B'"  %} 
    <h2 id="{{ year.name | slugify }}" class="archive__subtitle">{{ year.name }}</h2>
    {% for month in postsByMonth %}
        <h3 id="{{ month.name | slugify }}" class="archive__subtitle">{{ month.name }}</h2>
        {% for post in month.items %}
            {% include archive-single.html %}
        {% endfor %}
    {% endfor %}
{% endfor %}
{% endraw %}
```
연도별, 월별로 포스트를 가져와서 `archive-single.html`을 호출하는 코드다. `_includes/archive-single.html`을 구현한다.

```html
{% raw %}
{% if post.header.teaser %}
  {% capture teaser %}{{ post.header.teaser }}{% endcapture %}
{% else %}
  {% assign teaser = site.teaser %}
{% endif %}

{% if post.id %}
  {% assign title = post.title | markdownify | remove: "<p>" | remove: "</p>" %}
{% else %}
  {% assign title = post.title %}
{% endif %}

<div class="{{ include.type | default: "list" }}__item">
  <article class="archive__item" itemscope itemtype="http://schema.org/CreativeWork">
    {% if include.type == "grid" and teaser %}
      <div class="archive__item-teaser">
        <img src=
          {% if teaser contains "://" %}
            "{{ teaser }}"
          {% else %}
            "{{ teaser | absolute_url }}"
          {% endif %}
          alt="">
      </div>
    {% endif %}
    <h2 class="archive__item-title" itemprop="headline">
      {% if post.link %}
        <a href="{{ post.link }}">{{ title }}</a> <a href="{{ post.url | absolute_url }}" rel="permalink"><i class="fas fa-link" aria-hidden="true" title="permalink"></i><span class="sr-only">Permalink</span></a>
      {% else %}
        <a href="{{ post.url | absolute_url }}" rel="permalink">{{ title }}</a>
      {% endif %}
    </h2>
    {% if post.read_time %}
      <p class="page__meta"><i class="far fa-clock" aria-hidden="true"></i> {% include read-time.html %}</p>
    {% endif %}
    <!--
    {% if post.excerpt %}<p class="archive__item-excerpt" itemprop="description">{{ post.excerpt | markdownify | strip_html | truncate: 160 }}</p>{% endif %}
    -->
</article>
</div>
{% endraw %}
```
출처 : https://github.com/mmistakes/minimal-mistakes/blob/master/_includes/archive-single.html

이러면 archive기능 구현이 끝이다.

이상으로 댓글-disqus, google analytics, LaTex-MathJax, archive 구현에 대한 정리를 마친다.



