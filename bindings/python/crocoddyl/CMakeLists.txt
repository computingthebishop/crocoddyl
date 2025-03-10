FILE(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_BINDINGS_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
  )
FILE(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_BINDINGS_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  )

SET(${PROJECT_NAME}_PYTHON_BINDINGS_FILES
  __init__.py
  deprecated.py
  deprecation.py
  )

ADD_LIBRARY(${PROJECT_NAME}_pywrap SHARED ${${PROJECT_NAME}_PYTHON_BINDINGS_SOURCES} ${${PROJECT_NAME}_PYTHON_BINDINGS_HEADERS})
SET_TARGET_PROPERTIES(${PROJECT_NAME}_pywrap PROPERTIES SUFFIX ${PYTHON_EXT_SUFFIX})

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_pywrap ${PROJECT_NAME} eigenpy::eigenpy)
TARGET_LINK_BOOST_PYTHON(${PROJECT_NAME}_pywrap)
# BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS spews conversion warnings from int to long unsigned int.
# Unfortunately, using literals does not work in a macro. As such, this turns them off for the entire wrapper:
TARGET_COMPILE_OPTIONS(${PROJECT_NAME}_pywrap PRIVATE "-Wno-conversion")
# We are also exposing deprecated declarations until they are removed. Ignore the errors for the wrapper:
TARGET_COMPILE_OPTIONS(${PROJECT_NAME}_pywrap PRIVATE "-Wno-deprecated-declarations")

IF(UNIX AND NOT APPLE)
  SET_TARGET_PROPERTIES(${PROJECT_NAME}_pywrap PROPERTIES INSTALL_RPATH "\$ORIGIN/../../..")
ENDIF()

INSTALL(TARGETS ${PROJECT_NAME}_pywrap DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})

FOREACH(python ${${PROJECT_NAME}_PYTHON_BINDINGS_FILES})
  PYTHON_BUILD(. ${python})
  INSTALL(FILES
    "${${PROJECT_NAME}_SOURCE_DIR}/bindings/python/crocoddyl/${python}"
    DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})
ENDFOREACH(python ${${PROJECT_NAME}_PYTHON_BINDINGS_FILES})


SET(${PROJECT_NAME}_BINDINGS_UTILS_PYTHON_FILES
  __init__.py
  pendulum.py
  biped.py
  quadruped.py
  )

FOREACH(python ${${PROJECT_NAME}_BINDINGS_UTILS_PYTHON_FILES})
  PYTHON_BUILD(utils ${python})
  INSTALL(FILES
    "${${PROJECT_NAME}_SOURCE_DIR}/bindings/python/crocoddyl/utils/${python}"
    DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/utils)
ENDFOREACH(python ${${PROJECT_NAME}_BINDINGS_UTILS_PYTHON_FILES})
