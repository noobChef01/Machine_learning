"""
Main
-Capture the config file
-Create an agent instance
-Run the agent
date: Jan 2020
author: Sajid Mashroor
"""
import json
import argparse
import agents.nmt_attention 

def main():
    # parse the path of the json config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_pth', type=str,
                            default='None',
                            help='Path to Configuration file in json format')
    args = parser.parse_args()

    # with open(args.config_pth, "r") as json_file:
    #     config = json.load(json_file)

    with open("configs\\nmt_attention.json", "r") as json_file:
        config = json.load(json_file)
    
    agent_cls = getattr(agents.nmt_attention, config["agent"])
    agent = agent_cls(config)
    agent.run()
    agent.finalize()

if __name__ == '__main__':
    main()
