#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bmi.h"

#define NGIAB_NAME_CAPACITY 2048

typedef Bmi *(*register_bmi_cfe_function)(Bmi *model);

typedef struct {
    void *library;
    Bmi *model;
    int finalized;
} ngiab_cfe_handle;

static void set_error(char *buffer, size_t capacity, const char *message)
{
    if (buffer == NULL || capacity == 0) {
        return;
    }

    if (message == NULL) {
        message = "unknown CFE BMI bridge error";
    }

    snprintf(buffer, capacity, "%s", message);
}

static int require_handle(
    void *opaque,
    ngiab_cfe_handle **handle,
    char *error,
    size_t error_capacity)
{
    if (opaque == NULL) {
        set_error(error, error_capacity, "CFE handle is null");
        return 1;
    }

    *handle = (ngiab_cfe_handle *)opaque;

    if ((*handle)->model == NULL) {
        set_error(error, error_capacity, "CFE BMI model is null");
        return 1;
    }

    if ((*handle)->finalized) {
        set_error(error, error_capacity, "CFE BMI model is finalized");
        return 1;
    }

    return 0;
}

void *ngiab_cfe_open(
    const char *library_path,
    const char *configuration_path,
    char *error,
    size_t error_capacity)
{
    if (library_path == NULL || configuration_path == NULL) {
        set_error(
            error,
            error_capacity,
            "library_path and configuration_path are required");
        return NULL;
    }

    ngiab_cfe_handle *handle =
        (ngiab_cfe_handle *)calloc(1, sizeof(ngiab_cfe_handle));

    if (handle == NULL) {
        set_error(error, error_capacity, "failed to allocate CFE handle");
        return NULL;
    }

    handle->library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);

    if (handle->library == NULL) {
        set_error(error, error_capacity, dlerror());
        free(handle);
        return NULL;
    }

    dlerror();
    register_bmi_cfe_function register_model =
        (register_bmi_cfe_function)dlsym(
            handle->library,
            "register_bmi_cfe");
    const char *symbol_error = dlerror();

    if (symbol_error != NULL || register_model == NULL) {
        set_error(
            error,
            error_capacity,
            symbol_error == NULL
                ? "register_bmi_cfe symbol is unavailable"
                : symbol_error);
        dlclose(handle->library);
        free(handle);
        return NULL;
    }

    Bmi *allocated_model = (Bmi *)calloc(1, sizeof(Bmi));

    if (allocated_model == NULL) {
        set_error(
            error,
            error_capacity,
            "failed to allocate the CFE BMI function table");
        dlclose(handle->library);
        free(handle);
        return NULL;
    }

    handle->model = register_model(allocated_model);

    if (handle->model == NULL) {
        set_error(error, error_capacity, "register_bmi_cfe returned null");
        free(allocated_model);
        dlclose(handle->library);
        free(handle);
        return NULL;
    }

    if (handle->model != allocated_model) {
        set_error(
            error,
            error_capacity,
            "register_bmi_cfe returned an unexpected model pointer");
        free(allocated_model);
        dlclose(handle->library);
        free(handle);
        return NULL;
    }

    if (handle->model->initialize == NULL) {
        set_error(
            error,
            error_capacity,
            "CFE BMI initialize function pointer is null");
        free(handle->model);
        dlclose(handle->library);
        free(handle);
        return NULL;
    }

    int status =
        handle->model->initialize(handle->model, configuration_path);

    if (status != 0) {
        set_error(
            error,
            error_capacity,
            "CFE BMI initialize returned a failure status");
        free(handle->model);
        dlclose(handle->library);
        free(handle);
        return NULL;
    }

    set_error(error, error_capacity, "");
    return handle;
}

int ngiab_cfe_finalize(
    void *opaque,
    char *error,
    size_t error_capacity)
{
    if (opaque == NULL) {
        return 0;
    }

    ngiab_cfe_handle *handle = (ngiab_cfe_handle *)opaque;

    if (!handle->finalized && handle->model != NULL) {
        if (handle->model->finalize == NULL) {
            set_error(
                error,
                error_capacity,
                "CFE BMI finalize function pointer is null");
            return 1;
        }

        int status = handle->model->finalize(handle->model);
        if (status != 0) {
            set_error(
                error,
                error_capacity,
                "CFE BMI finalize returned a failure status");
            return status;
        }

        handle->finalized = 1;
    }

    if (handle->model != NULL) {
        free(handle->model);
        handle->model = NULL;
    }

    if (handle->library != NULL) {
        dlclose(handle->library);
        handle->library = NULL;
    }

    free(handle);
    set_error(error, error_capacity, "");
    return 0;
}

int ngiab_cfe_update(
    void *opaque,
    char *error,
    size_t error_capacity)
{
    ngiab_cfe_handle *handle = NULL;
    if (require_handle(
            opaque,
            &handle,
            error,
            error_capacity) != 0) {
        return 1;
    }

    if (handle->model->update == NULL) {
        set_error(
            error,
            error_capacity,
            "CFE BMI update function pointer is null");
        return 1;
    }

    int status = handle->model->update(handle->model);
    if (status != 0) {
        set_error(
            error,
            error_capacity,
            "CFE BMI update returned a failure status");
    }
    return status;
}

int ngiab_cfe_update_until(
    void *opaque,
    double time,
    char *error,
    size_t error_capacity)
{
    ngiab_cfe_handle *handle = NULL;
    if (require_handle(
            opaque,
            &handle,
            error,
            error_capacity) != 0) {
        return 1;
    }

    if (handle->model->update_until == NULL) {
        set_error(
            error,
            error_capacity,
            "CFE BMI update_until function pointer is null");
        return 1;
    }

    int status = handle->model->update_until(handle->model, time);
    if (status != 0) {
        set_error(
            error,
            error_capacity,
            "CFE BMI update_until returned a failure status");
    }
    return status;
}

static int get_count(
    void *opaque,
    int input,
    int *count,
    char *error,
    size_t error_capacity)
{
    ngiab_cfe_handle *handle = NULL;
    if (require_handle(
            opaque,
            &handle,
            error,
            error_capacity) != 0) {
        return 1;
    }

    if (count == NULL) {
        set_error(error, error_capacity, "count pointer is null");
        return 1;
    }

    int status;
    if (input) {
        if (handle->model->get_input_item_count == NULL) {
            set_error(
                error,
                error_capacity,
                "get_input_item_count function pointer is null");
            return 1;
        }
        status =
            handle->model->get_input_item_count(handle->model, count);
    }
    else {
        if (handle->model->get_output_item_count == NULL) {
            set_error(
                error,
                error_capacity,
                "get_output_item_count function pointer is null");
            return 1;
        }
        status =
            handle->model->get_output_item_count(handle->model, count);
    }

    if (status != 0) {
        set_error(
            error,
            error_capacity,
            input
                ? "get_input_item_count returned a failure status"
                : "get_output_item_count returned a failure status");
    }

    return status;
}

int ngiab_cfe_get_input_count(
    void *opaque,
    int *count,
    char *error,
    size_t error_capacity)
{
    return get_count(
        opaque,
        1,
        count,
        error,
        error_capacity);
}

int ngiab_cfe_get_output_count(
    void *opaque,
    int *count,
    char *error,
    size_t error_capacity)
{
    return get_count(
        opaque,
        0,
        count,
        error,
        error_capacity);
}

static void free_names(char **names, int count)
{
    if (names == NULL) {
        return;
    }

    for (int index = 0; index < count; ++index) {
        free(names[index]);
    }
    free(names);
}

static int get_name(
    void *opaque,
    int input,
    int index,
    char *buffer,
    size_t capacity,
    char *error,
    size_t error_capacity)
{
    int count = 0;
    int count_status = get_count(
        opaque,
        input,
        &count,
        error,
        error_capacity);

    if (count_status != 0) {
        return count_status;
    }

    if (index < 0 || index >= count) {
        set_error(
            error,
            error_capacity,
            "BMI variable-name index is outside the valid range");
        return 1;
    }

    if (buffer == NULL || capacity == 0) {
        set_error(
            error,
            error_capacity,
            "BMI variable-name output buffer is invalid");
        return 1;
    }

    ngiab_cfe_handle *handle = (ngiab_cfe_handle *)opaque;
    char **names = (char **)calloc((size_t)count, sizeof(char *));

    if (names == NULL) {
        set_error(
            error,
            error_capacity,
            "failed to allocate BMI variable-name array");
        return 1;
    }

    for (int position = 0; position < count; ++position) {
        names[position] =
            (char *)calloc(NGIAB_NAME_CAPACITY, sizeof(char));
        if (names[position] == NULL) {
            free_names(names, count);
            set_error(
                error,
                error_capacity,
                "failed to allocate BMI variable-name buffer");
            return 1;
        }
    }

    int status;
    if (input) {
        if (handle->model->get_input_var_names == NULL) {
            free_names(names, count);
            set_error(
                error,
                error_capacity,
                "get_input_var_names function pointer is null");
            return 1;
        }
        status =
            handle->model->get_input_var_names(handle->model, names);
    }
    else {
        if (handle->model->get_output_var_names == NULL) {
            free_names(names, count);
            set_error(
                error,
                error_capacity,
                "get_output_var_names function pointer is null");
            return 1;
        }
        status =
            handle->model->get_output_var_names(handle->model, names);
    }

    if (status == 0) {
        snprintf(buffer, capacity, "%s", names[index]);
    }
    else {
        set_error(
            error,
            error_capacity,
            input
                ? "get_input_var_names returned a failure status"
                : "get_output_var_names returned a failure status");
    }

    free_names(names, count);
    return status;
}

int ngiab_cfe_get_input_name(
    void *opaque,
    int index,
    char *buffer,
    size_t capacity,
    char *error,
    size_t error_capacity)
{
    return get_name(
        opaque,
        1,
        index,
        buffer,
        capacity,
        error,
        error_capacity);
}

int ngiab_cfe_get_output_name(
    void *opaque,
    int index,
    char *buffer,
    size_t capacity,
    char *error,
    size_t error_capacity)
{
    return get_name(
        opaque,
        0,
        index,
        buffer,
        capacity,
        error,
        error_capacity);
}

int ngiab_cfe_get_var_type(
    void *opaque,
    const char *name,
    char *buffer,
    size_t capacity,
    char *error,
    size_t error_capacity)
{
    ngiab_cfe_handle *handle = NULL;
    if (require_handle(
            opaque,
            &handle,
            error,
            error_capacity) != 0) {
        return 1;
    }

    if (name == NULL || buffer == NULL || capacity == 0) {
        set_error(
            error,
            error_capacity,
            "name and output buffer are required");
        return 1;
    }

    if (handle->model->get_var_type == NULL) {
        set_error(
            error,
            error_capacity,
            "get_var_type function pointer is null");
        return 1;
    }

    char type[NGIAB_NAME_CAPACITY] = {0};
    int status =
        handle->model->get_var_type(handle->model, name, type);

    if (status == 0) {
        snprintf(buffer, capacity, "%s", type);
    }
    else {
        set_error(
            error,
            error_capacity,
            "get_var_type returned a failure status");
    }

    return status;
}

int ngiab_cfe_get_var_units(
    void *opaque,
    const char *name,
    char *buffer,
    size_t capacity,
    char *error,
    size_t error_capacity)
{
    ngiab_cfe_handle *handle = NULL;
    if (require_handle(
            opaque,
            &handle,
            error,
            error_capacity) != 0) {
        return 1;
    }

    if (name == NULL || buffer == NULL || capacity == 0) {
        set_error(
            error,
            error_capacity,
            "name and output buffer are required");
        return 1;
    }

    if (handle->model->get_var_units == NULL) {
        set_error(
            error,
            error_capacity,
            "get_var_units function pointer is null");
        return 1;
    }

    char units[NGIAB_NAME_CAPACITY] = {0};
    int status =
        handle->model->get_var_units(handle->model, name, units);

    if (status == 0) {
        snprintf(buffer, capacity, "%s", units);
    }
    else {
        set_error(
            error,
            error_capacity,
            "get_var_units returned a failure status");
    }

    return status;
}

int ngiab_cfe_get_var_nbytes(
    void *opaque,
    const char *name,
    int *nbytes,
    char *error,
    size_t error_capacity)
{
    ngiab_cfe_handle *handle = NULL;
    if (require_handle(
            opaque,
            &handle,
            error,
            error_capacity) != 0) {
        return 1;
    }

    if (name == NULL || nbytes == NULL) {
        set_error(
            error,
            error_capacity,
            "name and nbytes pointer are required");
        return 1;
    }

    if (handle->model->get_var_nbytes == NULL) {
        set_error(
            error,
            error_capacity,
            "get_var_nbytes function pointer is null");
        return 1;
    }

    int status =
        handle->model->get_var_nbytes(handle->model, name, nbytes);

    if (status != 0) {
        set_error(
            error,
            error_capacity,
            "get_var_nbytes returned a failure status");
    }

    return status;
}

int ngiab_cfe_get_value_double(
    void *opaque,
    const char *name,
    double *value,
    char *error,
    size_t error_capacity)
{
    ngiab_cfe_handle *handle = NULL;
    if (require_handle(
            opaque,
            &handle,
            error,
            error_capacity) != 0) {
        return 1;
    }

    if (name == NULL || value == NULL) {
        set_error(
            error,
            error_capacity,
            "name and value pointer are required");
        return 1;
    }

    if (handle->model->get_value == NULL) {
        set_error(
            error,
            error_capacity,
            "get_value function pointer is null");
        return 1;
    }

    int nbytes = 0;
    int size_status =
        handle->model->get_var_nbytes(handle->model, name, &nbytes);

    if (size_status != 0 || nbytes != (int)sizeof(double)) {
        set_error(
            error,
            error_capacity,
            "requested BMI value is not one scalar double");
        return 1;
    }

    int status =
        handle->model->get_value(handle->model, name, value);

    if (status != 0) {
        set_error(
            error,
            error_capacity,
            "get_value returned a failure status");
    }

    return status;
}

int ngiab_cfe_set_value_double(
    void *opaque,
    const char *name,
    double value,
    char *error,
    size_t error_capacity)
{
    ngiab_cfe_handle *handle = NULL;
    if (require_handle(
            opaque,
            &handle,
            error,
            error_capacity) != 0) {
        return 1;
    }

    if (name == NULL) {
        set_error(error, error_capacity, "name is required");
        return 1;
    }

    if (handle->model->set_value == NULL) {
        set_error(
            error,
            error_capacity,
            "set_value function pointer is null");
        return 1;
    }

    int nbytes = 0;
    int size_status =
        handle->model->get_var_nbytes(handle->model, name, &nbytes);

    if (size_status != 0 || nbytes != (int)sizeof(double)) {
        set_error(
            error,
            error_capacity,
            "requested BMI value is not one scalar double");
        return 1;
    }

    int status =
        handle->model->set_value(handle->model, name, &value);

    if (status != 0) {
        set_error(
            error,
            error_capacity,
            "set_value returned a failure status");
    }

    return status;
}

static int get_time_value(
    void *opaque,
    int which,
    double *value,
    char *error,
    size_t error_capacity)
{
    ngiab_cfe_handle *handle = NULL;
    if (require_handle(
            opaque,
            &handle,
            error,
            error_capacity) != 0) {
        return 1;
    }

    if (value == NULL) {
        set_error(error, error_capacity, "time value pointer is null");
        return 1;
    }

    int status;
    if (which == 0) {
        if (handle->model->get_current_time == NULL) {
            set_error(
                error,
                error_capacity,
                "get_current_time function pointer is null");
            return 1;
        }
        status =
            handle->model->get_current_time(handle->model, value);
    }
    else if (which == 1) {
        if (handle->model->get_time_step == NULL) {
            set_error(
                error,
                error_capacity,
                "get_time_step function pointer is null");
            return 1;
        }
        status =
            handle->model->get_time_step(handle->model, value);
    }
    else {
        if (handle->model->get_end_time == NULL) {
            set_error(
                error,
                error_capacity,
                "get_end_time function pointer is null");
            return 1;
        }
        status =
            handle->model->get_end_time(handle->model, value);
    }

    if (status != 0) {
        set_error(
            error,
            error_capacity,
            "BMI time query returned a failure status");
    }

    return status;
}

int ngiab_cfe_get_current_time(
    void *opaque,
    double *value,
    char *error,
    size_t error_capacity)
{
    return get_time_value(
        opaque,
        0,
        value,
        error,
        error_capacity);
}

int ngiab_cfe_get_time_step(
    void *opaque,
    double *value,
    char *error,
    size_t error_capacity)
{
    return get_time_value(
        opaque,
        1,
        value,
        error,
        error_capacity);
}

int ngiab_cfe_get_end_time(
    void *opaque,
    double *value,
    char *error,
    size_t error_capacity)
{
    return get_time_value(
        opaque,
        2,
        value,
        error,
        error_capacity);
}
