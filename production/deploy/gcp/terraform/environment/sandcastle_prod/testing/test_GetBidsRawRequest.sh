# Setup arguments.
INPUT_PATH=get_bids_request.json  # Needs to be a valid GetBidsRawRequest
INPUT_FORMAT=JSON
BFE_HOST_ADDRESS=dsp-x-prod.bfe.ba.privacy-sandbox-demos-dsp-x.dev  # DNS name of BFE service (Example: dns:///buyer.domain.com)
#(For local runs services, use: BFE_HOST_ADDRESS=localhost:50051)
CLIENT_IP=142.251.175.95

# Run the tool with desired arguments.
./builders/tools/bazel-debian run //tools/secure_invoke:invoke \
    -- \
    -target_service=bfe \
    -input_file="/src/workspace/${INPUT_PATH}" \
    -input_format=${INPUT_FORMAT} \
    -host_addr=${BFE_HOST_ADDRESS} \
    -client_ip=${CLIENT_IP}
