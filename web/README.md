## React + Material UI Demo

**First**, install `node.js` and `npm` as you'd use **React** in Node environment.

Install **Homebrew** package manager.
```julia
$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
Update Homebrew to ensure it is up-to-date.
```julia
$ brew update
```
Now, install **Node** (**npm** will be installed with node):
```julia
$ brew install node
```
※ It is recommended that you'd keep **npm** version *up-to-date* !
```julia
$ npm install -g npm
```

**Then**, it's time to make a Global Package installation.

Required packages are: `babel`, `webpack` and `webpack-dev-server`.
```julia
$ npm install -g babel webpack webpack-dev-server
```
So it's all set to provide fundamental environment for React.

---

**Run Demo**

Clone the source from Git repo. For example,
```julia
$ git clone https://suisun_312@bitbucket.org/suisun_312/jukainlp_suisun_dev.git
```
Get to the demo's destination folder `web`.
```julia
$ cd web
```
Install dependencies for `node_modules` listed in **package.json**.
```julia
$ npm install
```
OK !

Finally, it's up to you for running **JukaiNLP** demo built with `React` and `Material UI`.
```julia
$ npm start
```
Open your browser and listen to **`localhost:7777`** which is running by `webpack-dev-server`.
