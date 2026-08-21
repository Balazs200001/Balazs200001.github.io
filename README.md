# Balazs200001.github.io

Source for my GitHub Pages blog, built with Jekyll and the
[midnight](https://github.com/pages-themes/midnight) remote theme.

## Structure

| Path | Purpose |
| --- | --- |
| `index.md` | Landing page, renders the card list of all posts (`_layouts/home.html`) |
| `_posts/` | One markdown file per article, named `YYYY-MM-DD-title.md` |
| `_layouts/` | `default.html` (shell), `home.html` (landing page), `post.html` (article) |
| `assets/posts/<post-slug>/` | Images and videos belonging to one post |
| `assets/css/custom.css` | Site styling layered on top of the remote theme |

## Running locally

Needs Ruby with the MSYS2 devkit (one-time):

```powershell
winget install RubyInstallerTeam.RubyWithDevKit.3.3
```

Then, in a fresh terminal:

```powershell
gem install bundler
bundle install
bundle exec jekyll serve --livereload
```

`jekyll-remote-theme` is held at 0.4.x, which is the version GitHub Pages builds with.
0.5.x depends on the `openssl` gem, which compiles against OpenSSL headers that
RubyInstaller's MSYS2 does not ship.

The site is served at <http://127.0.0.1:4000>. Edits to posts, layouts and CSS rebuild
automatically; `_config.yml` changes need a restart. The build fetches the remote theme,
so the first run needs an internet connection.

## Adding a post

Create `_posts/YYYY-MM-DD-some-title.md`:

```markdown
---
title: "Some Title"
date: 2026-01-15
tags: [Graphics, Real-Time Rendering]
description: "One or two sentences shown on the landing page card."
---

Post body starts here.
```

The `post` layout is applied automatically and the post appears on the landing page.
Post URLs are just the title, e.g. `/some-title/`.

Put that post's images and videos in `assets/posts/some-title/` and reference them with
an absolute path, e.g. `![Caption](/assets/posts/some-title/diagram.png)`. Relative paths
break because the page is served from a nested URL.

Tags are rendered as colored pills by `_includes/tag-list.html`, which turns each tag into
a `tag--<slug>` class. To give a new tag its own color, add `.tag--<slug> { --tag-h: <hue>; }`
to `assets/css/custom.css`; tags with no entry use the default hue.
