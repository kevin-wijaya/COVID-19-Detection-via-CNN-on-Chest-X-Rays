const PROD = JSON.parse(process.env.PROD_ENV || false);
console.log('\npostcss:', PROD ? 'production' : 'development', '\n')

let plugins = [
  require('tailwindcss'),  
  require('autoprefixer'),
]

if (PROD) {
  plugins.push(require('cssnano')({
    preset: 'default'
  }))
}

module.exports = {
  plugins: plugins
}