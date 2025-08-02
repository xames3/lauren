from concurrent.futures import ThreadPoolExecutor

import pytest

from lauren.core.config import _ALLOWED_LOG_LEVELS
from lauren.core.config import _DEFAULT_LOG_DATEFMT
from lauren.core.config import _DEFAULT_LOG_FMT
from lauren.core.config import Config
from lauren.core.config import ConsoleLoggerConfig
from lauren.core.config import FileLoggerConfig
from lauren.core.config import LoggerConfig
from lauren.core.config import config_property
from lauren.core.error import ConfigValidationError as Error


@pytest.fixture
def default():
    return config_property("mikasa")


@pytest.fixture
def allowed():
    return config_property("mikasa", allowed=("eren", "mikasa", "armin"))


@pytest.fixture
def between():
    return config_property(5, between=(1, 10))


@pytest.fixture
def frozen():
    return config_property("frozen_value", frozen=True)


@pytest.fixture
def mixed():
    return config_property(
        2,
        allowed=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        between=(1, 10),
        check=lambda x: x % 2 == 0,
    )


@pytest.fixture
def factory():
    def _create_test_class(name="internal", default=None, **kwargs):
        class TestClass:
            pass

        _property = config_property(default, **kwargs)
        _property.__set_name__(TestClass, name)
        setattr(TestClass, name, _property)
        return TestClass

    return _create_test_class


@pytest.mark.unit
class TestConfigProperty:
    @pytest.mark.parametrize(
        "default, frozen, description",
        [
            ("vegeta", True, "Prince of all Saiyans"),
            (9000, False, "Over 9000!"),
            ([], True, None),
        ],
    )
    def test_init_with_parameters(self, default, frozen, description):
        _property = config_property(
            default,
            frozen=frozen,
            description=description,
        )
        assert _property.default == default
        assert _property.frozen is frozen
        assert _property.description == description
        assert _property.allowed is None
        assert _property.check is None
        assert _property.between is None
        assert _property.property == ""
        assert _property.locks == {}

    def test_init_with_defaults(self):
        _property = config_property(None)
        assert _property.default is None
        assert _property.frozen is False
        assert _property.description is None
        assert _property.allowed is None
        assert _property.check is None
        assert _property.between is None
        assert _property.property == ""
        assert _property.validate is False
        assert _property.locks == {}

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("load", "_load"),
            ("max_connections", "_max_connections"),
            ("timeout", "_timeout"),
            ("retries", "_retries"),
        ],
    )
    def test_set_name_configures_name(self, name, expected, factory):
        TestClass = factory(name, "broly")
        property_descriptor = getattr(TestClass, name)
        assert property_descriptor.property == expected
        assert hasattr(TestClass, name)
        assert property_descriptor.default == "broly"

    @pytest.mark.parametrize(
        "allowed, valid, invalid",
        [
            (("aang", "katara", "sokka"), "sokka", "zuko"),
            ((-1, 1), -1, 0),
            ((True, False), False, None),
            (("DEBUG", "INFO", "ERROR"), "INFO", "TRACE"),
        ],
    )
    def test_set_value_with_validation(self, allowed, valid, invalid, factory):
        TestClass = factory("avatar", valid, allowed=allowed)
        instance = TestClass()
        instance.avatar = valid
        assert instance.avatar == valid
        with pytest.raises(Error, match="not one of the allowed values"):
            instance.avatar = invalid

    @pytest.mark.parametrize(
        "default, allowed, valid",
        [
            ("drigger", {"dragoon", "draciel", "drigger"}, "dragoon"),
            (3, {1, 2, 3}, 2),
            (False, {True, False}, True),
            ((3, 4), {frozenset([1, 2]), (3, 4)}, frozenset([1, 2])),
        ],
    )
    def test_validate_allowed_success(self, default, allowed, valid):
        _property = config_property(default, allowed=allowed)
        _property.__validate__(valid)

    @pytest.mark.parametrize(
        "default, allowed, invalid",
        [
            ("drigger", {"dragoon", "draciel", "drigger"}, "dranzer"),
            (3, {1, 2, 3}, 4),
            (True, {True, False}, "true"),
            ((3, 4), {frozenset([1, 2]), (3, 4)}, (0, 1)),
        ],
    )
    def test_validate_allowed_failure(self, default, allowed, invalid):
        _property = config_property(default, allowed=allowed)
        with pytest.raises(Error, match="not one of the allowed values"):
            _property.__validate__(invalid)

    @pytest.mark.parametrize(
        "default, between, valids",
        [
            (7, (1, 10), [1, 5, 10]),
            (0.69, (0.0, 1.0), [0.0, 0.5, 1.0]),
            (0, (-5, 5), [-5, 0, 5]),
        ],
    )
    def test_validate_between_success(self, default, between, valids):
        _property = config_property(default, between=between)
        for value in valids:
            _property.__validate__(value)

    @pytest.mark.parametrize(
        "default, between, invalids",
        [
            (7, (1, 10), [0, 11, 15]),
            (0.42, (0.0, 1.0), [-0.1, 1.1, 2.0]),
            (-3, (-5, 5), [-6, 6, 10]),
        ],
    )
    def test_validate_between_failure(self, default, between, invalids):
        _property = config_property(default, between=between)
        for value in invalids:
            with pytest.raises(Error, match="is not between"):
                _property.__validate__(value)

    @pytest.mark.parametrize(
        "allowed, invalid",
        [
            ({True, False}, "invalid"),
            ({1, "text", None}, "invalid"),
            ({frozenset([1, 2]), (3, 4), "string"}, "invalid"),
            ({1, 2.5, None, "string"}, "invalid"),
        ],
    )
    def test_error_message(self, allowed, invalid):
        _property = config_property(None, allowed=allowed)
        with pytest.raises(Error) as exc:
            _property.__validate__(invalid)
        assert "not one of the allowed values" in str(exc.value)

    def test_frozen(self, factory):
        TestClass = factory("spiderman", "peter parker", frozen=True)
        instance = TestClass()
        assert instance.spiderman == "peter parker"
        with pytest.raises(Error, match="cannot modify frozen property"):
            instance.spiderman = "miles morales"
        with pytest.raises(Error) as exc:
            instance.spiderman = "miles morales"
        assert "spiderman" in str(exc.value)

    @pytest.mark.parametrize(
        "default, expected",
        [
            (None, type(None)),
            (0, int),
            ("", str),
            ([], list),
            ({}, dict),
            (True, bool),
        ],
    )
    def test_property_descriptor_edge_cases(self, default, expected, factory):
        TestClass = factory("form", default)
        instance = TestClass()
        assert instance.form == default
        assert isinstance(instance.form, expected)
        new_value = "changed"
        instance.form = new_value
        assert instance.form == new_value
        instance.form = default
        assert instance.form == default

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "thread, iterations_per_thread",
        [
            (5, 100),
            (10, 200),
            (20, 500),
        ],
    )
    def test_thread_safety(self, thread, iterations_per_thread, factory):
        TestClass = factory(
            "member",
            "Superman",
            allowed={"Superman", "Batman", "Wonder Woman", "Flash", "Aquaman"},
        )
        jla = TestClass()
        results = []
        errors = []
        members = ["Superman", "Batman", "Wonder Woman", "Flash", "Aquaman"]

        def worker(wid):
            try:
                for index in range(iterations_per_thread):
                    member = members[index % len(members)]
                    jla.member = member
                    current = jla.member
                    results.append((wid, current))
                    if current not in members:
                        errors.append(
                            f"Invalid value {current} from worker {wid}"
                        )
            except Exception as e:
                errors.append(f"Worker {wid} error: {e}")

        with ThreadPoolExecutor(max_workers=thread) as executor:
            futures = [executor.submit(worker, i) for i in range(thread)]
            for future in futures:
                future.result()
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == thread * iterations_per_thread
        for wid, result in results:
            assert result in members, (
                f"Invalid result {result} from worker {wid}"
            )


@pytest.mark.integration
class TestFileLoggerConfig:
    @pytest.fixture
    def config(self):
        return FileLoggerConfig()

    def test_defaults(self, config):
        assert config.enable is True
        assert config.level == "INFO"
        assert config.fmt == _DEFAULT_LOG_FMT
        assert config.datefmt == _DEFAULT_LOG_DATEFMT
        assert config.path == "logs"
        assert config.output == "lauren.log"
        assert config.encoding == "utf-8"
        assert config.max_bytes == 10485760
        assert config.backups == 5

    @pytest.mark.parametrize("level", list(_ALLOWED_LOG_LEVELS))
    def test_level_allowed(self, config, level):
        config.level = level
        assert config.level == level

    @pytest.mark.parametrize("invalid", ["danger", "trace", "verbose"])
    def test_level_invalids(self, config, invalid):
        with pytest.raises(Error):
            config.level = invalid

    def test_encoding_frozen(self, config):
        with pytest.raises(Error, match="cannot modify frozen property"):
            config.encoding = "latin-1"


@pytest.mark.integration
class TestConsoleLoggerConfig:
    @pytest.fixture
    def config(self):
        return ConsoleLoggerConfig()

    def test_defaults(self, config):
        assert config.enable is True
        assert config.level == "DEBUG"
        assert config.fmt == _DEFAULT_LOG_FMT
        assert config.datefmt == _DEFAULT_LOG_DATEFMT
        assert config.colour is True


@pytest.mark.integration
class TestLoggerConfig:
    @pytest.fixture
    def config(self):
        return LoggerConfig()

    def test_defaults(self, config):
        assert config.level == "DEBUG"
        assert config.fmt == _DEFAULT_LOG_FMT
        assert config.datefmt == _DEFAULT_LOG_DATEFMT
        assert isinstance(config.file, FileLoggerConfig)
        assert isinstance(config.tty, ConsoleLoggerConfig)

    def test_nested_configuration_access(self, config):
        assert config.file.level == "INFO"
        assert config.tty.level == "DEBUG"


@pytest.mark.integration
class TestConfig:
    @pytest.fixture
    def config(self):
        return Config()

    def test_defaults(self, config):
        assert config.name == "lauren.core"
        assert config.version == "31.8.2025"
        assert isinstance(config.logger, LoggerConfig)

    @pytest.mark.parametrize(
        "frozen, new",
        [
            ("name", "new_name"),
            ("version", "new_version"),
        ],
    )
    def test_frozen_properties(self, config, frozen, new):
        with pytest.raises(Error, match="cannot modify frozen property"):
            setattr(config, frozen, new)

    def test_nested_logger_configuration(self, config):
        assert config.logger.level == "DEBUG"
        config.logger.level = "WARNING"
        assert config.logger.level == "WARNING"


@pytest.mark.extensive
class TestExtensiveValidation:
    @pytest.fixture
    def file(self):
        return FileLoggerConfig()

    @pytest.fixture
    def tty(self):
        return ConsoleLoggerConfig()

    @pytest.mark.parametrize("invalid", ["true", 2, "yes"])
    def test_file_logger_boolean_validation(self, file, invalid):
        with pytest.raises(Error):
            file.enable = invalid

    @pytest.mark.parametrize("invalid", ["debug", "Info", "error"])
    def test_file_logger_level_case_sensitivity(self, file, invalid):
        with pytest.raises(Error):
            file.level = invalid

    @pytest.mark.parametrize("backups", [0, 1, 5, 100, 1000])
    def test_file_logger_backups_valid_range(self, file, backups):
        file.backups = backups
        assert file.backups == backups

    @pytest.mark.parametrize(
        "max_bytes", [0, 1024, 1048576, 10485760, 104857600]
    )
    def test_file_logger_max_bytes_valid_range(self, file, max_bytes):
        file.max_bytes = max_bytes
        assert file.max_bytes == max_bytes

    @pytest.mark.parametrize("invalid", ["true", 2, "yes"])
    def test_console_logger_colour_validation(self, tty, invalid):
        with pytest.raises(Error):
            tty.colour = invalid

    @pytest.mark.parametrize("bool_equivalent", [(True, 1), (False, 0)])
    def test_console_logger_boolean_equivalence(self, tty, bool_equivalent):
        bool_value, int_value = bool_equivalent
        tty.colour = int_value
        assert tty.colour == int_value
        tty.colour = bool_value
        assert tty.colour is bool_value

    @pytest.mark.parametrize(
        "invalid", ["trace", "verbose", "info", "warn", "error", "fatal"]
    )
    def test_console_logger_invalid_levels(self, tty, invalid):
        with pytest.raises(Error):
            tty.level = invalid

    @pytest.mark.performance
    @pytest.mark.parametrize(
        "level",
        [
            pytest.param("light", marks=pytest.mark.fast),
            pytest.param("medium", marks=pytest.mark.slow),
            pytest.param("heavy", marks=pytest.mark.slow),
        ],
    )
    def test_validation_performance_stress(self, level):
        config_map = {
            "light": {"threads": 5, "iterations": 200},
            "medium": {"threads": 10, "iterations": 500},
            "heavy": {"threads": 20, "iterations": 1000},
        }
        config = config_map[level]

        class TestClass:
            _property = config_property("a", allowed={"a", "b", "c", "d", "e"})

        instance = TestClass()
        errors = []
        valids = ["a", "b", "c", "d", "e"]

        def worker():
            try:
                for index in range(config["iterations"]):
                    value = valids[index % len(valids)]
                    instance._property = value
                    current = instance._property
                    if current not in valids:
                        errors.append(f"Invalid value: {current}")
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=config["threads"]) as ex:
            futures = [ex.submit(worker) for _ in range(config["threads"])]
            for future in futures:
                future.result()
        assert len(errors) == 0, f"Errors in {level} stress test: {errors}"

    @pytest.mark.parametrize(
        "between, valids, invalids",
        [
            ((1, 10), [1, 5, 10], [0, 11]),
            ((0.0, 1.0), [0.0, 0.5, 1.0], [-0.1, 1.1]),
            ((-5, 5), [-5, 0, 5], [-6, 6]),
        ],
    )
    def test_between_validation_comprehensive(self, between, valids, invalids):
        _property = config_property(None, between=between)
        for value in valids:
            _property.__validate__(value)
        for value in invalids:
            with pytest.raises(Error, match="is not between"):
                _property.__validate__(value)

    @pytest.mark.parametrize(
        "invalid",
        [
            ("a", "z"),
            ([1, 2], [3, 4]),
            (None, 5),
        ],
    )
    def test_between_validation_invalid_ranges(self, invalid):
        _property = config_property(None, between=invalid)
        with pytest.raises(Error, match="must be a tuple of two numbers"):
            _property.__validate__(5)

    @pytest.mark.parametrize("num_loggers", [3, 5, 10])
    def test_config_instantiation_isolation(self, num_loggers):
        loggers = []
        allowed = list(_ALLOWED_LOG_LEVELS)
        for index in range(num_loggers):
            logger = LoggerConfig()
            logger.level = allowed[index % len(allowed)]
            loggers.append(logger)
        for index, logger in enumerate(loggers):
            expected = allowed[index % len(allowed)]
            assert logger.level == expected, (
                f"Logger {index} level mismatch: expected "
                f"{expected}, got {logger.level}"
            )

    def test_nested_config_independence(self):
        app_logger = LoggerConfig()
        app_logger.level = "ERROR"
        plugin_logger = LoggerConfig()
        assert plugin_logger.level == "DEBUG", (
            f"Expected DEBUG, got {plugin_logger.level}"
        )
        app_logger.level = "CRITICAL"
        plugin_logger.level = "WARNING"
        assert app_logger.level == "CRITICAL"
        assert plugin_logger.level == "WARNING"


if __name__ == "__main__":
    pytest.main([__file__])
