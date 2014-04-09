using ABC
using Distributions

include("test1.jl")
err_code = test1()
if err_code !=0 
  println("# Test 1 FAILED with error code ", err_code)
else
  println("# Test 1 passed.")
end

