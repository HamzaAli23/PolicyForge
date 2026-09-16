CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread -Iinclude

BUILD_DIR := build
CORE_SOURCES := src/gridworld.cpp src/algorithms.cpp src/report.cpp
APP_SOURCES := $(CORE_SOURCES) src/main.cpp
TEST_SOURCES := $(CORE_SOURCES) tests/test_main.cpp
APP := $(BUILD_DIR)/policyforge
TEST_APP := $(BUILD_DIR)/policyforge-tests

.PHONY: all test demo clean

all: $(APP)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(APP): $(APP_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(APP_SOURCES) -o $(APP)

$(TEST_APP): $(TEST_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_SOURCES) -o $(TEST_APP)

test: $(TEST_APP)
	./$(TEST_APP)

demo: $(APP)
	./$(APP) --environment examples/campus-navigation.env --output runs/demo

clean:
	rm -rf $(BUILD_DIR) runs
