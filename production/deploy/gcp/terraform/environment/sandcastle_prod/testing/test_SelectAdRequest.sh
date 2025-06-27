# copy this file and `select_ad_request.json` at the root of the `bidding-auction-servers `project.

# Setup arguments.
INPUT_PATH=select_ad_request.json  # Needs to be a valid plaintext in the root of the B&A project (i.e. the path is .../bidding-auction-server/select_ad_request.json)
SFE_HOST_ADDRESS=ssp-x-prod.sfe.ba.privacy-sandbox-demos-ssp-x.dev  # DNS name of SFE (e.g. dns:///seller.domain.com)
#(For local services, use: SFE_HOST_ADDRESS=localhost:50053)
CLIENT_IP=142.251.175.95

# Run the tool with desired arguments.
./builders/tools/bazel-debian run //tools/secure_invoke:invoke \
    -- \
    -target_service=sfe \
    -input_file="/src/workspace/${INPUT_PATH}" \
    -host_addr=${SFE_HOST_ADDRESS} \
    -client_ip=${CLIENT_IP}
