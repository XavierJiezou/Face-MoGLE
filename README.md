[![Python 3.8.12](https://img.shields.io/badge/webpage-template-green)](https://pages.github.com/)
![HTML5](https://img.shields.io/badge/html-orange?html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/css-blue?css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/javascript-yellow?javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
[![Bootstrap](https://img.shields.io/badge/bootstrap-purple?bootstrap-%238511FA.svg?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com/docs/5.0/getting-started/introduction/)

 
# Academic Webpage Template
 
This is a webpage template for academic papers. It was originally made for the [SinDDM](https://matankleiner.github.io/sinddm/) project page by [Matan Kleiner](https://www.linkedin.com/in/matan-kleiner/) with help from [Hila Manor](https://www.linkedin.com/in/hilamanor/). This template is a more general version that built in mind for the usage of other people. 

See an example of this academic webpage template [here.](https://matankleiner.github.io/academic-webpage-template/)

Examples of webpages using this template: 
* https://matankleiner.github.io/sinddm/
* https://inbarhub.github.io/DDPM_inversion/
* https://hilamanor.github.io/GaussianDenoisingPosterior/
* https://noamelata.github.io/NestedDiffusion/
* https://noa-cohen.github.io/MeaningfulDiversityInIR/

Suggestion comments and improvements are highly welcome, open an issue or contact me.    

### Usage 

To use this template, clone the repo: 
```
git clone https://github.com/matankleiner/academic-webpage-template.git
```
and follow the instruction in [github pages](https://pages.github.com/) to create your own webpage. 

This academic webpage template include the following components: 
* Navigation bar
* Scroll up button
* Dropdown menu for images, audio and video files
* Carousel for images, audio and video files
* Image slider
* Image and video overlay
* Before and after image slider
* Youtube embedding 
* Paper section with bibtex style citation and paper image

#### Change Style and Functionality
This academic webpage template include several components and functionality you might not need, comment them out or delete them from the `index.html` file. 
The functionality of some components is set in the `functionality.js` file (imported to `index.html`) while other is set via bootstrap class. 
The style of the webpage is mainly set in the `style.css` file. It includes, among others, font sizes, colors and position. However, not all style options are set in the `style.css` file, font sizes and position are sometime set in the `index.html` file and some styles and functionality are set as a bootstrap class.

Each component style and functionality is controlled in the following way: 
| <!-- -->    | <!-- -->    |
|-------------|-------------|
| Navigation bar  | bootstrap  |
| Scroll up button | js  |
| Dropdown  | bootstrap, js, css  |
| Carousel  | bootstrap, css  |
| Slider  | bootstrap, js, css  |
| Overlay  | css  |
| Before and after | js, css |
| Paper section  | css  |
| Text  | bootstrap, css  |
| Color  | bootstrap, css  |

An example for the academic webpage template in green can be found in the `index_green.html` and `style_green.css` files and [here](https://matankleiner.github.io/academic-webpage-template/index_green.html). 

Icons are taken from [font awesome](https://fontawesome.com/) or from [Academicons](https://jpswalsh.github.io/academicons/). For example, in order to change the [arrow up](https://fontawesome.com/icons/arrow-up?f=classic&s=solid) icon, use the font awesome interface. 

For the paper image, use an online pdf to png convertor, as this [one](https://pdf2png.com/).

Change the default shortcut icon, ```<link rel="shortcut icon" type="image/x-icon" href="resources/archive.svg">``` (part of the head), to your liking. Use `.svg` or `.ico` files and online [convertors](https://convertio.co/png-svg/) or [generators](https://www.favicon.cc/) to create them.

#### Bootstrap  
This webpage template is mainly built using [bootstrap 5](https://getbootstrap.com/docs/5.0/getting-started/introduction/). Bootstrap is a free open source framework that enable easy web development. Bootstrap class effect the style and functionality of different features. The [navigation bar](https://getbootstrap.com/docs/5.0/components/navbar/), [dropdown](https://getbootstrap.com/docs/5.3/components/dropdowns/), [carousel](https://getbootstrap.com/docs/5.3/components/carousel/) and [range](https://getbootstrap.com/docs/5.0/forms/range/) (for image slider) components style and functionality set by bootstrap. Another helpful option provided by bootstrap is the [grid system](https://getbootstrap.com/docs/5.3/layout/grid/).

### Acknowledgments 

This webpage was inspired by the template that was originally made for a [colorful ECCV paper](http://richzhang.github.io/colorization/). See how the original webpage looks [here](https://richzhang.github.io/webpage-template). 

The images in the webpage template are all generated by either DALL-E 2 or Stable Diffusion 2.1 Demo. The audio files are all generated by AudioLDM Demo and the video files are all generated by Text2Video-Zero Demo.

CSS and javascript might be taken from open source projects, credit is given in the `.css` or `.js` files. 

Badges in this `README.md` file were made with [Shields.io](https://shields.io/) or taken from [markdown-badges](https://github.com/Ileriayo/markdown-badges/tree/master). 
