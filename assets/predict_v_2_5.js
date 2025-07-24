const { spawn } = require("child_process");

function validateInput(input) {
  const requiredFields = [
    "brand",
    "mark",
    "Year_of_manufacture",
    "Year_of_entry",
    "Engine_capacity",
    "Engine",
    "Gearbox",
    "Hurd",
    "Drive",
    "Color",
    "Interior_color",
    "Conditions",
    "Mileage",
  ];

  for (const field of requiredFields) {
    if (
      !(field in input) ||
      input[field] === undefined ||
      input[field] === null
    ) {
      throw new Error(`Missing or invalid field: ${field}`);
    }
  }

  if (
    !Number.isInteger(input.Year_of_manufacture) ||
    input.Year_of_manufacture < 1900
  ) {
    throw new Error("Year_of_manufacture must be a valid integer year");
  }
  if (
    !Number.isInteger(input.Year_of_entry) ||
    input.Year_of_entry < input.Year_of_manufacture
  ) {
    throw new Error(
      "Year_of_entry must be a valid integer year >= Year_of_manufacture"
    );
  }

  if (typeof input.Mileage !== "number" && typeof input.Mileage !== "string") {
    throw new Error("Mileage must be a number or string");
  }
  if (typeof input.Mileage === "string") {
    if (!/^\d+$|^\d+-\d+$|^300000$/.test(input.Mileage)) {
      throw new Error(
        "Mileage string must be a number, range (e.g., '0-5000'), or '300000'"
      );
    }
  }

  const validConditions = ["00 гүйлттэй", "Дугаар авсан", "Дугаар аваагүй"];
  if (!validConditions.includes(input.Conditions)) {
    throw new Error(`Conditions must be one of: ${validConditions.join(", ")}`);
  }

  const categoricalFields = [
    "brand",
    "mark",
    "Engine_capacity",
    "Engine",
    "Gearbox",
    "Hurd",
    "Drive",
    "Color",
    "Interior_color",
  ];
  for (const field of categoricalFields) {
    if (typeof input[field] !== "string" || input[field].trim() === "") {
      throw new Error(`${field} must be a non-empty string`);
    }
  }

  return true;
}

function predictPrice(inputData) {
  return new Promise((resolve, reject) => {
    const env = Object.assign({}, process.env, { PYTHONIOENCODING: "utf-8" });
    const py = spawn("python", ["predict.py"], { env });
    let result = "";
    let errorOutput = "";

    const inputs = Array.isArray(inputData) ? inputData : [inputData];

    try {
      inputs.forEach((input, index) => {
        validateInput(input);
      });
    } catch (err) {
      return reject(new Error(`Input validation failed: ${err.message}`));
    }

    py.stdin.write(JSON.stringify(inputData));
    py.stdin.end();

    py.stdout.on("data", (data) => {
      result += data.toString();
    });

    py.stderr.on("data", (data) => {
      errorOutput += data.toString();
    });

    py.on("close", (code) => {
      if (code !== 0) {
        return reject(
          new Error(
            `Python process exited with code ${code}. Error: ${errorOutput}`
          )
        );
      }

      try {
        const parsedResult = JSON.parse(result);
        resolve(parsedResult);
      } catch (err) {
        reject(
          new Error(
            `Failed to parse Python output as JSON: ${err.message}. Output: ${result}`
          )
        );
      }
    });
  });
}

if (require.main === module) {
  const sampleInputs = [
    {
      brand: "Toyota", //output_with_app_columns.json
      mark: "Land Cruiser 300", //output_with_app_columns.json
      Year_of_manufacture: 2024, //заавал тоо
      Year_of_entry: 2024, //заавал тоо
      Engine_capacity: "3.6-4.0", //output_with_app_columns.json
      Engine: "Бензин", //select логик хийх. 2-3 төрөл бий
      Gearbox: "Автомат", //select логик хийх. 2-3 төрөл бий
      Hurd: "Зөв", //select логик хийх. 2 төрөл бий
      Drive: "Бүх дугуй 4WD", // select логик хийх
      Color: "Хар", // select логик хийх
      Interior_color: "Шаргал", // select логик хийх
      Conditions: "Дугаар авсан", // 3 төрлийн логик нөхцөл бий
      Mileage: "115000-120000", //Хэрэглэгчээс авсан утгын 0-5000, 5000-10000 гэж хөрвүүлэх логик хийгээрэй. Хэрвээ 5000 орж ирвэл 5000-10000, Хэрвээ 4999 гэж оруулбал 0-5000 гэж хөрвүүлэх. гэж мэтчилэн 500000 хүртэл хийнэ. median_data.xlsx загварыг харах
    },
  ];

  predictPrice(sampleInputs)
    .then((result) => {
      console.log("Predicted results:", JSON.stringify(result, null, 2));
    })
    .catch((err) => {
      console.error("Error:", err.message);
    });
}

module.exports = { predictPrice };
