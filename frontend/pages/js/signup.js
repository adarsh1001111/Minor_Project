const firebaseConfig = {
  apiKey: "AIzaSyBdyJhrePacqKikK3wIXFKzohg45vFLwGM",
  authDomain: "minor-a23b6.firebaseapp.com",
  projectId: "minor-a23b6",
  storageBucket: "minor-a23b6.appspot.com",
  messagingSenderId: "886563621851",
  appId: "1:886563621851:web:3d9f9fce40ea1a623035fe",
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Initialize Firebase Auth
const auth = firebase.auth();

// Google Authentication function
function signInWithGoogle() {
  var provider = new firebase.auth.GoogleAuthProvider();

  auth
    .signInWithPopup(provider)
    .then((result) => {
      const user = result.user;
      console.log("User signed in:", user);

      // Store user's name in localStorage
      localStorage.setItem("userName", user.displayName);

      // Redirect to index.html after successful login
      window.location.href = "index.html";
    })
    .catch((error) => {
      console.log("Error during sign-in:", error);
    });
}

// Add event listener to Google sign-up button
document.querySelector(".google-btn").addEventListener("click", function (e) {
  e.preventDefault(); // Prevent default link behavior
  signInWithGoogle(); // Call Google Sign-In function
});
