param(
    [string]$JobFile = "azureml/sales-retrain-job.yml"
)

$secureDatabricksPat = Read-Host "Databricks PAT" -AsSecureString
$databricksPatPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureDatabricksPat)

try {
    $databricksPat = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($databricksPatPtr)

    az ml job create `
        --file $JobFile `
        --set environment_variables.DATABRICKS_PAT=$databricksPat `
        --query "{name:name,status:status,studio_url:studio_url}" `
        -o json
}
finally {
    if ($databricksPatPtr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($databricksPatPtr)
    }

    Remove-Variable secureDatabricksPat -ErrorAction SilentlyContinue
    Remove-Variable databricksPat -ErrorAction SilentlyContinue
    Remove-Variable databricksPatPtr -ErrorAction SilentlyContinue
}