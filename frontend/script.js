console.log("JS loaded")

async function analyze(event){

    if(event){
        event.preventDefault()
    }

    let leafInput = document.getElementById("leaf")
    let pestInput = document.getElementById("pest")

    let leaf = leafInput.files[0]
    let pest = pestInput.files[0]

    if(!leaf || !pest){
        alert("Please upload both leaf and pest images")
        return
    }

    let formData = new FormData()

    formData.append("leaf", leaf)
    formData.append("pest", pest)

    try{

        document.getElementById("result").innerHTML = "<h3>Analyzing with AI...</h3>"

        let response = await fetch(
            "http://127.0.0.1:8000/analyze",
            {
                method: "POST",
                body: formData
            }
        )

        if(!response.ok){
            throw new Error("Server error")
        }

        let result = await response.json()

        document.getElementById("result").innerHTML = `
        <h3>Recommended Crop: ${result["Recommended Crop"]}</h3>
        <h3>Disease Detected: ${result["Disease Detected"]}</h3>
        <h3>Pest Detected: ${result["Pest Detected"]}</h3>
        <h3>Advice: ${result["Advice"]}</h3>
        `

    }
    catch(error){

        console.error(error)

        document.getElementById("result").innerHTML =
        "<h3 style='color:red;'>Error running AI analysis</h3>"
    }

}