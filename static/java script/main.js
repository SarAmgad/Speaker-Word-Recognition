let clicked =false
let source
let button=document.querySelector('.prediction_button')
let sign=document.querySelector('.right_wrong_sign')
let predictionLabel=document.querySelector('.prediction')
let predictionValue=predictionLabel.getAttribute('value')

let isRecognized=predictionValue.split('Other')==predictionValue
console.log(predictionValue)
// let array=predictionValue.split(' ')

function clicked_action(){

    if (clicked==true){
        source='url("../icons/noMic.png")'
        button.style.setProperty('--source',source)
        predictionLabel.innerHTML=''
        sign.style.visibility='hidden'
    }
    else {
        source='url("../icons/Mic.png")'
        button.style.setProperty('--source',source)
        sign.style.visibility='visible'
        if(!isRecognized){
            predictionLabel.innerHTML="You can't enter "
            source='url("../icons/Wrong_Sign.jpg")'
            sign.style.setProperty('--sign',source)
        }
        else {
            let array =predictionValue.split(' ')
            console.log(array)
            predictionLabel.innerHTML='Welcome'+' '+array[0]
            source='url("../icons/Right_Sign.jpg")'
            sign.style.setProperty('--sign',source)
        }

    }


    clicked=!clicked

}
// setTimeout(5000)


button.addEventListener('click',clicked_action())