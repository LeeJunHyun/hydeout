---
layout: post
title: '[Web] Jekyll을 이용해서 개인 Github Page 만들기'
categories:
  - Coding
tags:
  - web
  - ruby
  - jekyll
  - html
  - tutorial
---

Reference : https://jekyllrb-ko.github.io/

Github.io에서 제공하는 호스팅 서비스를 이용해서 정적 웹 페이지를 만드는 것을 정리한 포스트입니다.

---
---

## Jekyll이란?
Jekyll은 ruby 기반의 정적 사이트 생성기이다. 손쉽게 블로그를 만들 수 있고, [GitHub Pages](https://pages.github.com/)의 내부엔진이기때문에 Jekyll로 페이지를 만들어 개인블로그나 웹사이트를 github 서버에 무료로 호스팅 할 수 있다. 이미 만들어진 테마도 많으니 [인터넷](http://jekyllthemes.org/)에 올려진 테마를 받아서 사용하면 편리하다.
이 포스트는 https://github.com/fongandrew/hydeout에서 받은 https://fongandrew.github.io/hydeout/ 테마를 기준으로 정리되었다.

![Jekyll Theme](/assets/img/Coding/jekyll-theme.png)
*http://jekyllthemes.org/*


## github에서 페이지 호스팅하기
[GitHub](https://github.com/) 에서 각자의 레포를 *name*.github.io로 생성하면, https://*name*.github.io 라는 도메인으로 자동 호스팅 해준다. 

예를들어, 내 페이지의 github repository는 https://github.com/LeeJunHyun/LeeJunHyun.github.io 이고, https://leejunhyun.github.io 라는 도메인으로 호스팅되고있다. **이때, *name*은 github의 본인 name과 같아야 한다.**

새로 만든 레포에 index.html을 만들어 아무 내용이나 써놓고 push해보고 https://*name*.github.io 로 접속해보면 호스팅 되고 있다는 것을 확인 할 수 있다. (push 하고나서 호스팅까지 시간이 몇분 소요될 수도 있다.)


## localhost로 Jekyll 돌리기 
(수정사항이 페이지에 반영되기까지 몇분정도 소요되므로) 계속 github에 push해가며 페이지를 수정할 수 없으니, localhost로 돌려놓고 확정된 것만 push하기위해 로컬에 Jekyll을 설치한다.

우선, [여기](https://rubyinstaller.org/downloads/)에서 루비를 설치한다. 

```bash
$ ruby -v

ruby 2.3.1p112 (2016-04-26) [x86_64-linux-gnu]
```

또한 플러그인이나 테마를 사용하기 위해 bundler도 설치한다.
```bash
$ gem install bundler
$ gem install jekyll
```
permission 관련 에러는 sudo권한을 통해 해결한다.

이미 만들어진 테마를 github에 올려놓았다면
```bash
$ git clone https://github.com/name/name.github.io
$ cd name.github.io
```
처음부터 만든다면
```bash
$ mkdir jekyll-new-site
$ cd jekyll-new-site
```
으로 폴더를 만들어준다.

```bash
$ bundle init
# Gemfile 생성, 기존에 Gemfile이 있다면 지우고 다시 생성한다.

$ bundle add jekyll
# Jekyll 추가

$ bundle install
# 루비 젬 설치

$ jekyll serve
# 로컬호스트 서버 실행
```
이제 127.0.0.1:4000에서 서버가 돌아가고 있음을 확인 할 수 있다.

jekyll theme를 받아서 사용하는 경우 무언가 플러그인이 없다는 오류가 발생한다면, 그에 맞는 것을 install 해주면 된다. 로컬에 설치된 gem 목록은 다음과 같이 확인한다.

```bash
$ gem list
```

혹시 gem list에 있는데 다음과 같은 오류가 발생한다면,
![jekyll-bundle-error](/assets/img/Coding/jekyll-bundle-error.png)

각각 theme에서 사용되는 플러그인을 추가해준다.
```bash
$ bundle add jekyll-feed
```

각자의 theme에서 사용되는 플러그인은 `_config.yml` 에서 확인할 수 있다.

```yml
plugins:
  - jekyll-feed
  - jekyll-gist
  - jekyll-paginate
```

## Theme 수정하기
`_config.yml` 에는 환경설정 정보들이 들어가있다.
여기서 기본적인 설정들을 바꾸거나 다른 곳에서 사용할 변수들을 추가해줄 수 있다.

```yml
url:              https://fongandrew.github.io
baseurl:          '/hydeout'
```
우선 이 정보들을 아래처럼 수정해준다.

```yml
url:              https://name.github.io
# 본인페이지 기본 주소
baseurl:          ''
```
나머지는 본인 설정에 맞게 바꾸어주면 된다.

`_includes`에는 재사용되는 소스들을 넣어준다. 예를 들어, sidebar을 구현해놓은 sidebar.html을 `_includes/sidebar.html` 에 넣어주고 다른 파일에서 갖다쓸 때는 {% raw %}`{% include sidebar.html %}`{% endraw %}
을 해주면 된다.

`_posts`에는 컨텐츠가 저장된다. 페이지에 글을 올리고 싶다면 여기에 글을 저장하면 된다. 본 포스트는 `_posts/2018-09-01-Github-page.md` 로 저장되어있다. 포스트 이름은 `yyyy-mm-dd-your-posts-name.md` 형식으로 작성하면 된다.

`_layouts`에는 템플릿들을 넣어준다. 각 포스트별로 레이아웃의 템플릿을 지정해주면 여기 넣어둔 템플릿을 사용할 수 있다. 예를 들어, 본 포스트의 레이아웃은 `_layouts/post.html`이며 포스트(`_posts/2018-09-01-Github-page.md`) 상단에 다음과같이 지정해준다.
```yml
---
layout: post
title: '[Web] Jekyll을 이용해서 개인 Github Page 만들기'
categories:
  - Coding
tags:
  - web
  - ruby
  - jekyll
  - html
  - tutorial
---
```

내가 사용하는 테마에서 글들의 카테고리는 `category`에 저장된다.
본 포스트의 카테고리는 `Coding`이고, `category/coding.md`로 저장되어있다.
```yml
---
layout: category
title: Coding
comments: true
---

about Implementation
```

`about.md`에는 소개를 넣으면 된다.


이상으로 정리를 마친다. 추가적으로 댓글기능 `disqus` 를 넣거나 `google analytics`를 설정하는 것은 테마에 미리 구현되어 있어서 약간의 수정만으로 편리하게 적용 할 수 있다. 글들을 모아보는 `archive`기능은 테마에 원래 없었어서 약간은 불편하게 수정했는데, 이러한 내용에 대해서는 다음에 LaTex문법 적용하는 방법과 함께 정리를 할 예정이다.

