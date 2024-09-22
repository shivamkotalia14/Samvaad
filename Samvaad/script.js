// Changing the style of scroll bar
// window.onscroll = function() {myFunction()};
		
// function myFunction() {
// 	var winScroll = document.body.scrollTop || document.documentElement.scrollTop;
// 	var height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
// 	var scrolled = (winScroll / height) * 100;
// 	document.getElementById("myBar").style.width = scrolled + "%"; 
// }

function scrollAppear() {
  var introText = document.querySelector('.side-text');
  var sideImage = document.querySelector('.sideImage');
  var introPosition = introText.getBoundingClientRect().top;
  var imagePosition = sideImage.getBoundingClientRect().top;

  var screenPosition = window.innerHeight / 1.2;

  if (introPosition < screenPosition) {
      introText.classList.add('side-text-appear');
  }
  if (imagePosition < screenPosition) {
      sideImage.classList.add('sideImage-appear');
  }
}

window.addEventListener('scroll', scrollAppear);

// For switching between navigation menus in mobile mode
var i = 2;
function switchTAB() {
  var x = document.getElementById("list-switch");
  if (i % 2 == 0) {
      x.style = "display: grid; height: 50vh; margin-left: 5%;";
      document.getElementById("search-switch").style = "display: block; margin-left: 5%;";
  } else {
      x.style = "display: none;";
      document.getElementById("search-switch").style = "display: none;";
  }
  i++;
}

// For LOGIN
var x = document.getElementById("login");
var y = document.getElementById("register");
var z = document.getElementById("btn");
var a = document.getElementById("log");
var b = document.getElementById("reg");
var w = document.getElementById("other");

function register() {
  x.style.left = "-400px";
  y.style.left = "50px";
  z.style.left = "110px";
  w.style.visibility = "hidden";
  b.style.color = "#fff";
  a.style.color = "#000";
}

function login() {
  x.style.left = "50px";
  y.style.left = "450px";
  z.style.left = "0px";
  w.style.visibility = "visible";
  a.style.color = "#fff";
  b.style.color = "#000";
}

// CheckBox Function
function goFurther() {
  if (document.getElementById("chkAgree").checked) {
      document.getElementById('btnSubmit').style = 'background: linear-gradient(to right, #FA4B37, #DF2771);';
  } else {
      document.getElementById('btnSubmit').style = 'background: lightgray;';
  }
}

function google() {
  window.location.assign("https://accounts.google.com/signin/v2/identifier?service=accountsettings&continue=https%3A%2F%2Fmyaccount.google.com%2F%3Futm_source%3Dsign_in_no_continue&csig=AF-SEnbZHbi77CbAiuHE%3A1585466693&flowName=GlifWebSignIn&flowEntry=AddSession", "_blank");
}

// Function to submit data to the backend
async function submitData(data) {
  try {
      const response = await fetch('/your-endpoint-url', { // Replace with your backend endpoint
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify(data),
      });

      if (!response.ok) {
          throw new Error('Network response was not ok');
      }

      const responseData = await response.json();
      // Process the response data (e.g., display it in the frontend)
      displayResponse(responseData);

  } catch (error) {
      console.error('There was a problem with the fetch operation:', error);
  }
}

// Function to display response on the frontend
function displayResponse(data) {
  // Example: append data to a specific element
  const responseContainer = document.getElementById('response-container'); // Ensure this element exists in your HTML
  responseContainer.innerHTML += `<p>${data.message}</p>`; // Customize based on your response structure
}

// Example submission handler
document.getElementById('your-submit-button').addEventListener('click', function() {
  const inputData = { /* Collect your data here, e.g., from form inputs */ };
  submitData(inputData);
});
